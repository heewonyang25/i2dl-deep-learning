import os
from zipfile import ZipFile


RELEVANT_FOLDERS = ['models', 'exercise_code']


def zipdir(path, ziph):
    """ Recursively adds a folder and all its subfolders to a zipfile
    :param path: path of input folder to be added to zipfile
    :param ziph: a ZipFile object
    """
    # print(path)
    # print(os.walk(path))
    for root, dirs, files in os.walk(path):
        for file in files:
            # print(file)
            ziph.write(os.path.join(root, file))


def submit_exercise(
    zip_output_filename='submission',
    data_path='.',
    relevant_folders=RELEVANT_FOLDERS
):
    """ Creates a curated zip out of submission related files
    :param zip_output_filename: output filename of zip without extension
    :param data_path: path where we look for required files
    :param relevant_folder: folders which we consider for zipping besides
    jupyter notebooks
    """
    # Notebook filenames
    notebooks_filenames = [x for x in os.listdir(data_path)
                           if x.endswith('.ipynb')]
    # Existing relevant folders
    relevant_folders = [x for x in os.listdir(data_path)
                        if x in relevant_folders]
    print('relevant folders: {}\nnotebooks files: {}'.format(
        relevant_folders, notebooks_filenames))

    # Check output filename
    if not zip_output_filename.endswith('.zip'):
        zip_output_filename += '.zip'

    # Create output directory if the student removed it
    folder_path = os.path.dirname(zip_output_filename)
    if folder_path != '':
        os.makedirs(folder_path, exist_ok=True)

    with ZipFile(zip_output_filename, 'w') as myzip:
        # Add relevant folders
        for folder in relevant_folders:
            print('Adding folder {}'.format(folder))
            myzip.write(folder)
            zipdir(folder, myzip)
        # Add notebooks
        for fn in notebooks_filenames:
            print('Adding notebook {}'.format(fn))
            myzip.write(fn)

    print('Zipping successful! Zip is stored under: {}'.format(
        os.path.abspath(zip_output_filename)
    ))
