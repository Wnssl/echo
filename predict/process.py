import pydicom
import torch
from skimage import transform
import numpy as np
import cv2


def get_preprocessed_tensor(dicom_file_path, fps=None, hr=None, orientation="Mayo", masked=True):
    min_number_of_frames = 30
    # Defining the range of acceptable heart rate values
    min_hr = 30
    max_hr = 150
    dicom_dataset = pydicom.dcmread(dicom_file_path, force=True)
    fps = None
    hr = None
    orientation = "Mayo"
    if hasattr(dicom_dataset, "NumberOfFrames"):
        if dicom_dataset.NumberOfFrames < 2:
            raise ValueError("DICOM file has <2 frames!")
    else:
        raise AttributeError("No NumberOfFrames DICOM tag!")

    # Ensuring that the DICOM file does not have color Doppler
    if hasattr(dicom_dataset, "UltrasoundColorDataPresent"):
        if dicom_dataset.UltrasoundColorDataPresent:
            raise ValueError("DICOM file with color Doppler!")

    # Ensuring that the DICOM file contains only one ultrasound region
    if hasattr(dicom_dataset, "SequenceOfUltrasoundRegions"):
        if len(dicom_dataset.SequenceOfUltrasoundRegions) > 1:
            raise ValueError("DICOM file contains more than 1 US regions!")

    # Extracting heart rate from DICOM tags if not provided by the user
    if hr is None:
        if not hasattr(dicom_dataset, "HeartRate"):
            raise ValueError("Heart rate was not found in DICOM tags!")
        else:
            hr = dicom_dataset.HeartRate

    # Checking whether heart rate falls into the predefined range
    if hr < min_hr or hr > max_hr:
        raise ValueError("Heart rate falls outside of the predefined range ({} - {}/min)".format(min_hr, max_hr))

    # Extracting frame rate from DICOM tags if not provided by the user
    if fps is None:
        if hasattr(dicom_dataset, "RecommendedDisplayFrameRate"):
            fps = dicom_dataset.RecommendedDisplayFrameRate
        elif hasattr(dicom_dataset, "FrameTime"):
            fps = round(1000 / float(dicom_dataset.FrameTime))
        else:
            raise ValueError("Frame rate was not found in DICOM tags!")

    num_of_frames = dicom_dataset.NumberOfFrames
    len_of_cardiac_cycle = (60 / int(hr)) * int(float(fps))
    gray_frames = dicom_dataset.pixel_array[:, :, :, 0]
    if orientation == "Stanford":
        for i, frame in enumerate(gray_frames):
            gray_frames[i] = cv2.flip(frame, 1)

    shape_of_frames = gray_frames.shape
    changes = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    changes_frequency = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    binary_mask = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    cropped_frames = []

    for i in range(len(gray_frames) - 1):
        diff = abs(gray_frames[i] - gray_frames[i + 1])
        changes += diff
        nonzero = np.nonzero(diff)
        changes_frequency[nonzero[0], nonzero[1]] += 1
    max_of_changes = np.amax(changes)
    min_of_changes = np.amin(changes)

    for r in range(len(changes)):
        for p in range(len(changes[r])):
            if int(changes_frequency[r][p]) < 10:
                changes[r][p] = 0
            else:
                changes[r][p] = int(255 * ((changes[r][p] - min_of_changes) / (max_of_changes - min_of_changes)))

    nonzero_values_for_binary_mask = np.nonzero(changes)

    binary_mask[nonzero_values_for_binary_mask[0], nonzero_values_for_binary_mask[1]] += 1
    kernel = np.ones((5, 5), np.int32)
    erosion_on_binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
    # image_show(erosion_on_binary_mask)
    binary_mask_after_erosion = np.where(erosion_on_binary_mask, binary_mask, 0)
    # image_show(binary_mask_after_erosion)
    nonzero_values_after_erosion = np.nonzero(binary_mask_after_erosion)
    binary_mask_coordinates = np.array([nonzero_values_after_erosion[0], nonzero_values_after_erosion[1]]).T

    cropped_mask = binary_mask_after_erosion[
                   np.min(binary_mask_coordinates[:, 0]):np.max(binary_mask_coordinates[:, 0]),
                   np.min(binary_mask_coordinates[:, 1]):np.max(binary_mask_coordinates[:, 1])]

    for row in cropped_mask:
        ids = [i for i, x in enumerate(row) if x == 1]
        if len(ids) < 2:
            continue
        row[ids[0]:ids[-1]] = 1

    for i in range(len(gray_frames)):
        masked_image = np.where(erosion_on_binary_mask, gray_frames[i], 0)

        cropped_image = masked_image[np.min(binary_mask_coordinates[:, 0]):np.max(binary_mask_coordinates[:, 0]),
                        np.min(binary_mask_coordinates[:, 1]):np.max(binary_mask_coordinates[:, 1])]
        cropped_frames.append(cropped_image)

    resized_frames = []
    for frame in cropped_frames:
        resized_frame = transform.resize(frame, (224, 224))
        resized_frames.append(resized_frame)
    resized_frames = np.asarray(resized_frames)
    resized_binary_mask = transform.resize(cropped_mask, (224, 224))

    frames_3ch = []
    for frame in resized_frames:
        new_frame = np.zeros((np.array(frame).shape[0], np.array(frame).shape[1], 3))
        new_frame[:, :, 0] = frame
        new_frame[:, :, 1] = frame
        new_frame[:, :, 2] = frame
        frames_3ch.append(new_frame)
    frames_tensor = np.array(frames_3ch)
    frames_tensor = frames_tensor.transpose((0, 3, 1, 2))
    binary_mask_tensor = np.array(resized_binary_mask)
    frames_tensor = torch.from_numpy(frames_tensor)
    binary_mask_tensor = torch.from_numpy(binary_mask_tensor)

    f, c, h, w = frames_tensor.size()
    new_shape = (f, 3, h, w)
    expanded_frames = frames_tensor.expand(new_shape)
    expanded_frames_clone = expanded_frames.clone()
    if masked:
        expanded_frames_clone[:, 0, :, :] = binary_mask_tensor
    else:
        pass
    return expanded_frames_clone