import os
import numpy as np
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt

def generate_coordinate_matrix(coord_system):
    """
    Generate a transformation matrix from the given coordinate system
    to the world coordinate system.
    :param coord_system: String representing the coordinate system (e.g., 'LSP', 'RAS', etc.)
    :return: 4x4 transformation matrix
    """
    if len(coord_system) != 3:
        raise ValueError("Coordinate system must have exactly 3 characters (e.g., 'RAS', 'LSP').")
    
    # Mapping from coordinate labels to axis directions
    axis_mapping = {
        'R': [1, 0, 0],   # Right (positive X)
        'L': [-1, 0, 0],  # Left (negative X)
        'A': [0, 1, 0],   # Anterior (positive Y)
        'P': [0, -1, 0],  # Posterior (negative Y)
        'S': [0, 0, 1],   # Superior (positive Z)
        'I': [0, 0, -1]   # Inferior (negative Z)
    }

    # Extract the directions for X, Y, Z
    x_axis = axis_mapping[coord_system[0]]
    y_axis = axis_mapping[coord_system[1]]
    z_axis = axis_mapping[coord_system[2]]

    # Ensure the coordinate system is valid 
    if not np.isclose(np.dot(x_axis, y_axis), 0) or not np.isclose(np.dot(x_axis, z_axis), 0) or not np.isclose(np.dot(y_axis, z_axis), 0):
        raise ValueError(f"Invalid coordinate system: {coord_system}. Axes must be orthogonal.")
    
    # Create the transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, 0] = x_axis  # X axis
    transform_matrix[:3, 1] = y_axis  # Y axis
    transform_matrix[:3, 2] = z_axis  # Z axis

    return transform_matrix

def calculate_transform_matrix(source_coord, target_coord):
    """
    Calculate the transformation matrix to convert coordinates
    from source_coord to target_coord.
    :param source_coord: Source coordinate system (e.g., 'LSP')
    :param target_coord: Target coordinate system (e.g., 'RAS')
    :return: 4x4 transformation matrix
    """
    # Generate matrices for both systems
    source_to_world = generate_coordinate_matrix(source_coord)
    target_to_world = generate_coordinate_matrix(target_coord)

    # Calculate the transform from source to target
    transform_matrix = np.linalg.inv(target_to_world) @ source_to_world
    return transform_matrix


    
def compute_affine(dicom_files):
    dicom_files = sorted(
            [files for files in dicom_files],
            key=lambda x: pydicom.dcmread(x).InstanceNumber
        )
    
    ds = pydicom.dcmread(dicom_files[0])
    
    image_orientation = np.array(ds.ImageOrientationPatient) 
    #print(f'image_orientation: {image_orientation}')
    #print(f'image_position: {ds.ImagePositionPatient}')
    row_cosine = image_orientation[3:]
    col_cosine = image_orientation[:3]
    #col_cosine = image_orientation[:3]
    #row_cosine = image_orientation[3:]
    pixel_spacing = np.array(ds.PixelSpacing) 
    
    first_position = np.array(ds.ImagePositionPatient)  
    
    if len(dicom_files) > 1:
        ds_next = pydicom.dcmread(dicom_files[1])
        second_position = np.array(ds_next.ImagePositionPatient)
        #print(f'first_instance_number: {ds.InstanceNumber}')
        #print(f'second_instance_number: {ds_next.InstanceNumber}')

        #print(f'first_position: {first_position}')
        #print(f'second_position: {second_position}')
        
        slice_direction = second_position - first_position
        #print(f'slice_direction before: {slice_direction}')
        slice_spacing = np.linalg.norm(slice_direction)
        slice_cosine = slice_direction / slice_spacing

    else:
        slice_cosine = np.cross(col_cosine,row_cosine)
        slice_spacing = 1.0
    
    #print(f'slice_spacing: {slice_spacing}')

    affine = np.eye(4)
    affine[:3, 0] = row_cosine * pixel_spacing[0]
    affine[:3, 1] = col_cosine * pixel_spacing[1]
    affine[:3, 2] = slice_cosine * slice_spacing* np.sign(ds_next.InstanceNumber-ds.InstanceNumber)
    affine[:3, 3] = first_position

    
    return affine

def flip_by_axes(nii_Data, dicom_val, nii_val):
    for i in range(3):
        print(dicom_val[i] , nii_val[i])
        if dicom_val[i] * nii_val[i] < 0:
            nii_Data = np.flip(nii_Data, axis=i)
    return nii_Data





def find_sort_order(list1, list2):

    indices = [ list1.index(poi) for poi in list2]
    

    return indices
def save_overlay_and_mask(dicom_files, reoriented_nifti, output_overlay_dir, output_mask_dir):
    if not os.path.exists(output_overlay_dir):
        os.makedirs(output_overlay_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    if not os.path.exists(output_raw_dir):
        os.makedirs(output_raw_dir)

    mask_mapping = []

    for i, dicom_file in enumerate(dicom_files):
        ds = pydicom.dcmread(dicom_file)
        original_image = ds.pixel_array

        # Resize the reoriented NIfTI mask to match the DICOM resolution
        mask_slice = reoriented_nifti[:, :, i]

        # Create overlay by combining original image and mask
        overlay = np.stack([original_image, original_image, original_image], axis=-1)
        overlay[mask_slice > 0] = [255, 0, 0]  # Red for mask

        # Save overlay as JPG
        overlay_filename = os.path.join(output_overlay_dir, f'{os.path.splitext(os.path.basename(dicom_file))[0]}.jpg')
        overlay = (overlay / overlay.max() * 255).astype(np.uint8)


        plt.imsave(overlay_filename, overlay, cmap='gray')

        # Save mask as PNG
        mask_filename = os.path.join(output_mask_dir, f'{os.path.splitext(os.path.basename(dicom_file))[0]}.png')
        plt.imsave(mask_filename, mask_slice, cmap='gray')

        # Append to mask mapping
        mask_mapping.append({'DICOM': dicom_file, 'Mask': mask_filename})
        #print (f'saving to {mask_filename}')

    return mask_mapping

import csv

def main(dcm_folder,nii_file,out_folder):
    
    dataset,basename =  dcm_folder.strip("/").split("/")[-3],dcm_folder.strip("/").split("/")[-1]
    print(f'Processing {dataset} {basename}')
    output_overlay_dir = os.path.join(out_folder,f'{dataset}',f'{basename}','overlay') 
    output_mask_dir = os.path.join(out_folder,f'{dataset}',f'{basename}','mask') 

    output_raw_dir = os.path.join(out_folder,f'{dataset}',f'{basename}','raw') 

    mapping_path = os.path.join(out_folder,f'{dataset}','mapping')
    output_csv_path = os.path.join(mapping_path,f'{basename}.csv') 
    

    dicom_files = sorted(
        [os.path.join(dcm_folder, f) for f in os.listdir(dcm_folder) if f.endswith('.dcm')],
        key=lambda x: pydicom.dcmread(x).InstanceNumber
    )
    try:
        nii_img = nib.load(nii_file)
    except:
        print(f'Segmentation {nii_file} not found')
        return 0
    nii_data = nii_img.get_fdata()
    nii_affine = nii_img.affine

    dicom_affine = compute_affine(dicom_files)
    print (f'Dicom affine matrix {dicom_affine}')
    print (f'nifti affine matrix {nii_affine}')
    m_lps2ras = calculate_transform_matrix('RAS', 'LPS')
    m_nifti2dicom = np.linalg.inv(m_lps2ras @ dicom_affine) @ nii_affine
    ornt = nib.orientations.io_orientation(m_nifti2dicom)
    reoriented_nifti = nib.orientations.apply_orientation(nii_data, ornt)
    mask_mapping = save_overlay_and_mask(dicom_files, reoriented_nifti, output_overlay_dir, output_mask_dir, raw_reoriented_nifti, output_raw_dir)




    if not os.path.exists(mapping_path):
        os.makedirs(mapping_path)
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['DICOM', 'Mask'])
        writer.writeheader()
        writer.writerows(mask_mapping)

if __name__ == '__main__':
    dcm_folders = os.listdir(r'dcm_path')
    dcm_folders = [os.path.join(r'dcmfolders',poi) for poi in dcm_folders]
    out_folder = r''
    for dcm_folder in dcm_folders:
        nii_file = f'output/{os.path.basename(dcm_folder)}/pancreas.nii.gz'
        main(dcm_folder,nii_file,out_folder,raw_nifti_data)
