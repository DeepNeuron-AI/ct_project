from os.path import dirname, join
from time import time, sleep
import multiprocessing

import pydicom
from pydicom.data import get_testdata_files
from pydicom.filereader import read_dicomdir

import numpy as np
from scipy.ndimage import zoom

from utils import clean_and_threshold, pad_height, get_pixels_hu

# fetch the path to the test data
flesh_threshold = -200
bone_threshold = 220
correctedHeight = 550
zoom_scaling = .125

SAVEPATH = "/path/to/save/"
LOADPATH = "/path/to/load/"
DICOMDIRS_PATHS = [] # Complete with DICOM directories


# Read the DICOM directories
dicom_dirs = []
for path in DICOMDIRS_PATHS:
    dicom_dirs.append({"dicom": read_dicomdir(LOADPATH + path + "/DICOMDIR"), "path": LOADPATH + path})

# find the maximum height in the scans
maxHeight = 0
scans = []

for d in dicom_dirs:
    for patient_record in d["dicom"].patient_records:
        for study in patient_record.children:
            for series in study.children:
                if len(series.children) > 10:
                    maxHeight = max(maxHeight, len(series.children))
                    scans.append({'path': d['path'], 'series': series})

print("Max height of a scan:", maxHeight)
print("Number of scans:", len(scans))


# the function that actual processes each scan
def saveStudy(scan, i):
    series = scan['series']
    p = scan['path']

    flesh = []
    bone = []
    raw = []

    print("===>", i)

    for image in series.children:
        # Convert to numpy
        # CHANGE THE FILE PATH
        path = join(p, *image.ReferencedFileID)
        dcm = pydicom.read_file(path)
        flesh_pixels = get_pixels_hu(dcm)
        bone_pixels = get_pixels_hu(dcm)

        # Filter out air / bone
        flesh_clean = clean_and_threshold(flesh_pixels, flesh_threshold)
        bone_clean = clean_and_threshold(bone_pixels, bone_threshold)

        # Append slice
        raw.append(dcm.pixel_array)
        flesh.append(flesh_clean)
        bone.append(bone_clean)

    # Save raws
    np.save(SAVEPATH + "raw/" + str(i) + ".npy", raw)

    # Zoom, Pad to cube and save xs
    flesh = zoom(np.asarray(flesh), zoom_scaling)
    flesh = pad_height(flesh, int(correctedHeight*zoom_scaling))
    np.save(SAVEPATH + "flesh/" + str(i) + ".npy", flesh)

    # Zoom, Pad to cube and save ys
    bone = zoom(np.asarray(bone), zoom_scaling)
    bone = pad_height(bone, int(correctedHeight*zoom_scaling))
    np.save(SAVEPATH + "bone/" + str(i) + ".npy", bone)

    print(i, "<===")


# multiprocessing code starts here

# create a job for each scan and start it
jobs = []
queue = []
nw = 40

# Add all jobs to queue
for i, scan in enumerate(scans):
    p = multiprocessing.Process(target=saveStudy, args=(scans[i], i))
    queue.append(p)

# Start first batch of jobs
for i in range(nw):
    p = queue.pop(0)
    jobs.append(p)
    p.start()

# loop here until all the jobs have finished
start = time()
check_again = True

while check_again:
    check_again = False

    # Iterate over active jobs
    for i, job in enumerate(jobs):
        if job.is_alive():
            check_again = True

        # If job has finished, add next from queue
        else:
            p = queue.pop(0)
            jobs.append(p)
            p.start()

    # Clean up finished jobs
    jobs = [x for x in jobs if x.is_alive()]

    sleep(1)

end = time()

print('time elapsed:', end - start)
