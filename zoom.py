import json
import multiprocessing as mp
from argparse import Namespace
import os
from pathlib import Path
import re
from print import print, printProgress

import numpy
import psutil
from PIL import Image
from turbojpeg import TurboJPEG

MAX_QUALITY = False  		# Set this to true if you want to compress/postprocess the images yourself later

QUALITY = 80
BG_QUALITY = 95

EXT = ".png"
OUT_EXT = ".webp"     		# format='JPEG' is hardcoded in places, meed to modify those, too. Most parameters are not supported outside jpeg.
THUMBNAIL_EXT = ".png"
BG_EXT = ".png"

BACKGROUND_COLOR = (0, 0, 0, 0)
THUMBNAIL_SCALE = 2

MIN_RENDERBOX_SIZE = 8


BG_IN_SIZE = 512
CHUNKS_PER_BG = 8
BG_OUT_SIZE = CHUNKS_PER_BG * 32 * 2 # 2 pixels per tile


# note that these are all 64 bit libraries since factorio doesnt support 32 bit.
if os.name == "nt":
    jpeg = TurboJPEG(Path(__file__, "..", "mozjpeg/turbojpeg.dll").resolve().as_posix())
# elif _platform == "darwin":						# I'm not actually sure if mac can run linux libraries or not.
# 	jpeg = TurboJPEG("mozjpeg/libturbojpeg.dylib")	# If anyone on mac has problems with the line below please make an issue :)
else:
    jpeg = TurboJPEG(Path(__file__, "..", "mozjpeg/libturbojpeg.so").resolve().as_posix())


def saveCompress(img, path: Path):
    # if MAX_QUALITY:  # do not waste any time compressing the image
    #     return img.save(path, subsampling=0, quality=100)

    # outFile = path.open("wb")
    # outFile.write(jpeg.encode(numpy.array(img)[:, :, ::-1].copy()))
    # outFile.close()
    img.save(path, "webp")


def simpleZoom(workQueue):
    for (folder, start, stop, filename) in workQueue:
        path = Path(folder, str(start), filename)
        img = Image.open(path.with_suffix(EXT), mode="r").convert("RGBA")
        if OUT_EXT != EXT:
            saveCompress(img, path.with_suffix(OUT_EXT))
            path.with_suffix(EXT).unlink()

        for z in range(start - 1, stop - 1, -1):
            if img.size[0] >= MIN_RENDERBOX_SIZE * 2 and img.size[1] >= MIN_RENDERBOX_SIZE * 2:
                img = img.resize((img.size[0] // 2, img.size[1] // 2), Image.Resampling.LANCZOS)
            zFolder = Path(folder, str(z))
            if not zFolder.exists():
                zFolder.mkdir(parents=True)
            saveCompress(img, Path(zFolder, filename).with_suffix(OUT_EXT))


def zoomRenderboxes(daytimeSurfaces, toppath, timestamp, subpath, args):
    with Path(toppath, "mapInfo.json").open("r+", encoding="utf-8") as mapInfoFile:
        mapInfo = json.load(mapInfoFile)

        outFile = Path(toppath, "mapInfo.out.json")
        if outFile.exists():
            with outFile.open("r", encoding="utf-8") as mapInfoOutFile:
                outInfo = json.load(mapInfoOutFile)
        else:
            outInfo = {"maps": {}}

        mapLayer = None
        mapIndex = None

        for i, m in enumerate(mapInfo["maps"]):
            if m["path"] == timestamp:
                mapLayer = m
                mapIndex = str(i)

        if not mapLayer or not mapIndex:
            raise Exception("mapLayer or mapIndex missing")

        if mapIndex not in outInfo["maps"]:
            outInfo["maps"][mapIndex] = {"surfaces": {}}

        zoomWork = set()
        for daytime, activeSurfaces in daytimeSurfaces.items():
            surfaceZoomLevels = {}
            for surfaceName in activeSurfaces:
                surfaceZoomLevels[surfaceName] = (
                    mapLayer["surfaces"][surfaceName]["zoom"]["max"]
                    - mapLayer["surfaces"][surfaceName]["zoom"]["min"]
                )

            for surfaceName, surface in mapLayer["surfaces"].items():
                if "links" in surface:

                    if surfaceName not in outInfo["maps"][mapIndex]["surfaces"]:
                        outInfo["maps"][mapIndex]["surfaces"][surfaceName] = {}
                    if "links" not in outInfo["maps"][mapIndex]["surfaces"][surfaceName]:
                        outInfo["maps"][mapIndex]["surfaces"][surfaceName]["links"] = []

                    for linkIndex, link in enumerate(surface["links"]):
                        if link["type"] == "link_renderbox_area" and "zoom" in link:
                            totalZoomLevelsRequired = 0
                            for zoomSurface, zoomLevel in link["maxZoomFromSurfaces"].items():
                                if zoomSurface in surfaceZoomLevels:
                                    totalZoomLevelsRequired = max(
                                        totalZoomLevelsRequired,
                                        zoomLevel + surfaceZoomLevels[zoomSurface],
                                    )

                            if not outInfo["maps"][mapIndex]["surfaces"][surfaceName]["links"][linkIndex]:
                                outInfo["maps"][mapIndex]["surfaces"][surfaceName]["links"][linkIndex] = {}
                            if "zoom" not in outInfo["maps"][mapIndex]["surfaces"][surfaceName]["links"][linkIndex]:
                                outInfo["maps"][mapIndex]["surfaces"][surfaceName]["links"][linkIndex]["zoom"] = {}

                            link["zoom"]["min"] = link["zoom"]["max"] - totalZoomLevelsRequired
                            outInfo["maps"][mapIndex]["surfaces"][surfaceName]["links"][linkIndex]["zoom"]["min"] = link["zoom"]["min"]

                            # an assumption is made that the total zoom levels required doesnt change between snapshots.
                            if (link if "path" in link else outInfo["maps"][mapIndex]["surfaces"][surfaceName]["links"][linkIndex])["path"] == timestamp:
                                zoomWork.add(
                                    (
                                        Path(
                                            subpath,
                                            mapLayer["path"],
                                            link["toSurface"],
                                            daytime if link["daynight"] else "day",
                                            "renderboxes",
                                        ).resolve(),
                                        link["zoom"]["max"],
                                        link["zoom"]["min"],
                                        link["filename"],
                                    )
                                )

        with outFile.open("w", encoding="utf-8") as mapInfoOutFile:
            json.dump(outInfo, mapInfoOutFile)
            mapInfoOutFile.truncate()

    maxthreads = args.zoomthreads if args.zoomthreads else args.maxthreads
    processes = []
    zoomWork = list(zoomWork)
    for i in range(0, min(maxthreads, len(zoomWork))):
        p = mp.Process(target=simpleZoom, args=(zoomWork[i::maxthreads],))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def work(basepath, pathList, surfaceName, daytime, size, start, stop, last, chunk, keepLast=False):
    chunksize = 2 ** (start - stop)
    if start > stop:
        for k in range(start, stop, -1):
            x = chunksize * chunk[0]
            y = chunksize * chunk[1]
            for j in range(y, y + chunksize, 2):
                for i in range(x, x + chunksize, 2):

                    coords = [(0, 0), (1, 0), (0, 1), (1, 1)]
                    paths = [
                        Path(
                            basepath,
                            pathList[0],
                            surfaceName,
                            daytime,
                            str(k),
                            str(i + coord[0]),
                            str(j + coord[1]),
                        ).with_suffix(EXT)
                        for coord in coords
                    ]

                    if any(path.exists() for path in paths):

                        if not Path(basepath, pathList[0], surfaceName, daytime, str(k - 1), str(i // 2)).exists():
                            try:
                                Path(basepath, pathList[0], surfaceName, daytime, str(k - 1), str(i // 2)).mkdir(parents=True)
                            except OSError:
                                pass

                        isOriginal = []
                        for m in range(len(coords)):
                            isOriginal.append(paths[m].is_file())
                            if not isOriginal[m]:
                                for n in range(1, len(pathList)):
                                    paths[m] = Path(basepath, pathList[n], surfaceName, daytime, str(k), str(i + coords[m][0]), str(j + coords[m][1])).with_suffix(OUT_EXT)
                                    if paths[m].is_file():
                                        break

                        result = Image.new("RGBA", (size, size), BACKGROUND_COLOR)

                        images = []
                        for m in range(len(coords)):
                            if paths[m].is_file():
                                img = Image.open(paths[m], mode="r").convert("RGBA")
                                result.paste(
                                    box=(
                                        coords[m][0] * size // 2,
                                        coords[m][1] * size // 2,
                                    ),
                                    im=img.resize(
                                        (size // 2, size // 2), Image.Resampling.LANCZOS
                                    ),
                                )

                                if isOriginal[m]:
                                    images.append((img, paths[m]))

                        if k == last + 1:
                            saveCompress(result, Path(basepath, pathList[0], surfaceName, daytime, str(k - 1), str(i // 2), str(j // 2)).with_suffix(OUT_EXT))
                        if OUT_EXT != EXT and (k != last + 1 or keepLast):
                            result.save(Path(basepath, pathList[0], surfaceName, daytime, str(k - 1), str(i // 2), str(j // 2), ).with_suffix(EXT))

                        if OUT_EXT != EXT:
                            for img, path in images:
                                saveCompress(img, path.with_suffix(OUT_EXT))
                                path.unlink()

            chunksize = chunksize // 2
    elif stop == last:
        path = Path(basepath, pathList[0], surfaceName, daytime, str(start), str(chunk[0]), str(chunk[1]))
        img = Image.open(path.with_suffix(EXT), mode="r").convert("RGB")
        saveCompress(img, path.with_suffix(OUT_EXT))
        path.with_suffix(EXT).unlink()


def thread(basepath, pathList, surfaceName, daytime, size, start, stop, last, allChunks, counter, resultQueue, keepLast=False):
    #print(start, stop, chunks)
    while True:
        with counter.get_lock():
            i = counter.value - 1
            if i < 0:
                return
            counter.value = i
        chunk = allChunks[i]
        work(basepath, pathList, surfaceName, daytime, size, start, stop, last, chunk, keepLast)
        resultQueue.put(True)




def zoom(
    outFolder: Path,
    timestamp: str = None,
    surfaceReference: str = None,
    daytimeReference: str = None,
    basepath: Path = None,
    needsThumbnail: bool = True,
    args: Namespace = Namespace(),
):

    psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)

    workFolder = basepath if basepath else Path(__file__, "..", "..", "..", "script-output", "FactorioMaps").resolve()

    topPath = Path(workFolder, outFolder)
    dataPath = Path(topPath, "mapInfo.json")
    imagePath = Path(topPath, "Images")
    maxthreads = args.zoomthreads if args.zoomthreads else args.maxthreads

    with dataPath.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for mapIndex, map in enumerate(data["maps"]):
        if timestamp is None or map["path"] == timestamp:
            for surfaceName, surface in map["surfaces"].items():
                if surfaceReference is None or surfaceName == surfaceReference:
                    maxzoom = surface["zoom"]["max"]
                    minzoom = surface["zoom"]["min"]

                    daytimes = []
                    if "day" in surface:
                        daytimes.append("day")
                    if "night" in surface:
                        daytimes.append("night")
                    for daytime in daytimes:
                        if daytimeReference is None or daytime == daytimeReference:
                            if not Path(topPath, "Images", str(map["path"]), surfaceName, daytime, str(maxzoom - 1)).is_dir():
                                printProgress("zoom", 0)

                                generateThumbnail = (
                                    needsThumbnail
                                    and mapIndex == len(data["maps"]) - 1
                                    and surfaceName
                                    == (
                                        "nauvis"
                                        if "nauvis" in map["surfaces"]
                                        else sorted(map["surfaces"].keys())[0]
                                    )
                                    and daytime == daytimes[0]
                                )

                                allBigChunks = {}
                                minX = float("inf")
                                maxX = float("-inf")
                                minY = float("inf")
                                maxY = float("-inf")
                                imageSize: int = None
                                for xStr in Path(imagePath, str(map["path"]), surfaceName, daytime, str(maxzoom)).iterdir():
                                    x = int(xStr.name)
                                    minX = min(minX, x)
                                    maxX = max(maxX, x)
                                    for yStr in Path(imagePath, str(map["path"]), surfaceName, daytime, str(maxzoom), xStr).iterdir():
                                        if imageSize is None:
                                            imageSize = Image.open(Path(imagePath, str(map["path"]), surfaceName, daytime, str(maxzoom), xStr, yStr), mode="r").size[0]
                                        y = int(yStr.stem)
                                        minY = min(minY, y)
                                        maxY = max(maxY, y)
                                        allBigChunks[
                                            (
                                                x >> maxzoom-minzoom,
                                                y >> maxzoom-minzoom,
                                            )
                                        ] = True

                                if len(allBigChunks) <= 0:
                                    continue

                                pathList = []
                                for otherMapIndex in range(mapIndex, -1, -1):
                                    pathList.append(str(data["maps"][otherMapIndex]["path"]))

                                threadsplit = 0
                                while 4**threadsplit * len(allBigChunks) < maxthreads:
                                    threadsplit = threadsplit + 1
                                threadsplit = min(max(maxzoom - minzoom - 3, 0), threadsplit + 3)
                                allChunks = []
                                for pos in list(allBigChunks):
                                    for i in range(2**threadsplit):
                                        for j in range(2**threadsplit):
                                            allChunks.append(
                                                (
                                                    pos[0] * (2**threadsplit) + i,
                                                    pos[1] * (2**threadsplit) + j,
                                                )
                                            )

                                threads = min(len(allChunks), maxthreads)
                                processes = []
                                originalSize = len(allChunks)

                                # print(("%s %s %s %s" % (pathList[0], str(surfaceName), daytime, pathList)))
                                # print(("%s-%s (total: %s):" % (start, stop + threadsplit, len(allChunks))))
                                counter = mp.Value("i", originalSize)
                                resultQueue = mp.Queue()
                                for _ in range(0, threads):
                                    p = mp.Process(
                                        target=thread,
                                        args=(
                                            imagePath,
                                            pathList,
                                            surfaceName,
                                            daytime,
                                            imageSize,
                                            maxzoom,
                                            minzoom + threadsplit,
                                            minzoom,
                                            allChunks,
                                            counter,
                                            resultQueue,
                                            generateThumbnail,
                                        ),
                                    )
                                    p.start()
                                    processes.append(p)

                                doneSize = 0
                                for _ in range(originalSize):
                                    resultQueue.get(True)
                                    doneSize += 1
                                    printProgress("zoom", float(doneSize) / originalSize * 0.98)

                                for p in processes:
                                    p.join()

                                if threadsplit > 0:
                                    # print(("finishing up: %s-%s (total: %s)" % (stop + threadsplit, stop, len(allBigChunks))))
                                    processes = []
                                    i = len(allBigChunks) - 1
                                    for chunk in list(allBigChunks):
                                        p = mp.Process(
                                            target=work,
                                            args=(
                                                imagePath,
                                                pathList,
                                                surfaceName,
                                                daytime,
                                                imageSize,
                                                minzoom + threadsplit,
                                                minzoom,
                                                minzoom,
                                                chunk,
                                                generateThumbnail,
                                            ),
                                        )
                                        i = i - 1
                                        p.start()
                                        processes.append(p)
                                    for p in processes:
                                        p.join()

                                if generateThumbnail:
                                    print("generating thumbnail")
                                    minzoompath = Path(
                                        imagePath,
                                        str(map["path"]),
                                        surfaceName,
                                        daytime,
                                        str(minzoom),
                                    )

                                    if imageSize is None:
                                        raise Exception("Missing imageSize for thumbnail generation")

                                    thumbnail = Image.new(
                                        "RGB",
                                        (
                                            (maxX - minX + 1) * imageSize >> maxzoom-minzoom,
                                            (maxY - minY + 1) * imageSize >> maxzoom-minzoom,
                                        ),
                                        BACKGROUND_COLOR,
                                    )
                                    bigMinX = minX >> maxzoom-minzoom
                                    bigMinY = minY >> maxzoom-minzoom
                                    xOffset = ((bigMinX * imageSize << maxzoom-minzoom) - minX * imageSize) >> maxzoom-minzoom
                                    yOffset = ((bigMinY * imageSize << maxzoom-minzoom) - minY * imageSize) >> maxzoom-minzoom
                                    for chunk in list(allBigChunks):
                                        path = Path(minzoompath, str(chunk[0]), str(chunk[1])).with_suffix(EXT)
                                        thumbnail.paste(
                                            box=(
                                                xOffset + (chunk[0] - bigMinX) * imageSize,
                                                yOffset + (chunk[1] - bigMinY) * imageSize,
                                            ),
                                            im=Image.open(path, mode="r")
                                            .convert("RGB")
                                            .resize((imageSize, imageSize), Image.Resampling.LANCZOS),
                                        )

                                        if OUT_EXT != EXT:
                                            path.unlink()

                                    thumbnail.save(Path(imagePath, "thumbnail" + THUMBNAIL_EXT))

                                printProgress("zoom", 1, True)



                            # TODO: MULTITHREAD THIS
                            printProgress("bg", 0)

                            bgImages = {}
                            for xStr in Path(imagePath, str(map["path"]), surfaceName, daytime, "bg").iterdir():
                                x = int(xStr.name)
                                for yStr in Path(imagePath, str(map["path"]), surfaceName, daytime, str(maxzoom), xStr).iterdir():
                                    m = re.match(r'^(-?\d+)_(\d+)_(\d+)$', yStr.stem, re.IGNORECASE)
                                    assert(m)
                                    y = int(m.group(1))
                                    subX = int(m.group(2))
                                    subY = int(m.group(3))

                                    key = (x, y)
                                    if key not in bgImages:
                                        bgImages[key] = []
                                    bgImages[key].append((subX, subY))

                            doneCount = 0
                            for key, subImages in bgImages.items():
                                result = Image.new("RGB", (BG_IN_SIZE * CHUNKS_PER_BG, BG_IN_SIZE * CHUNKS_PER_BG), BACKGROUND_COLOR)
                                for subX, subY in subImages:
                                    path = Path(imagePath, str(map["path"]), surfaceName, daytime, "bg", str(key[0]), f"{key[1]}_{subX}_{subY}{EXT}")
                                    img = Image.open(path, mode="r").convert("RGB")
                                    result.paste(
                                        box=(
                                            subX * BG_IN_SIZE,
                                            subY * BG_IN_SIZE,
                                        ),
                                        im=img,
                                    )
                                    path.unlink()

                                result = result.resize((BG_OUT_SIZE, BG_OUT_SIZE), Image.Resampling.LANCZOS)
                                result.save(Path(imagePath, str(map["path"]), surfaceName, daytime, "bg", f"{key[0]}/{key[1]}{BG_EXT}"))

                                doneCount += 1
                                printProgress("bg", float(doneCount) / len(bgImages))
                            printProgress("bg", 1, True)
