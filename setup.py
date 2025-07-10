from setuptools import setup, find_packages

# Authors and their corresponding emails
AUTHORS = [
    ("Lalith Kumar Shiyam Sundar", "lalith.shiyamsundar@meduniwien.ac.at"),
    ("Sebastian Gutschmayer", "sebastian.gutschmayer@meduniwien.ac.at"),
    ("Manuel Pires", "manuel.pires@meduniwien.ac.at"),
]

# Convert the authors to a formatted string
authors_string = ", ".join([name for name, email in AUTHORS])
emails_string = ", ".join([email for name, email in AUTHORS])

setup(
    name='lionz',
    version='0.18.0',
    packages=find_packages(),
    install_requires=[
        'nnunetv2>=2.6.0',
        'nibabel',
        'halo',
        'pandas',
        'SimpleITK',
        'pydicom~=2.2.2',
        'argparse',
        'imageio',
        'numpy<2.0',
        'matplotlib',
        'pyfiglet',
        'natsort',
        'pillow',
        'colorama',
        'dask',
        'rich',
        'dicom2nifti~=2.4.8',
        'emoji',
        'dask[distributed]',
        'opencv-python',
    ],

    entry_points={
        'console_scripts': [
            'lionz=lionz.lionz:main'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    description='A toolkit for precise segmentation of tumors in PET/CT scans.',
    long_description='''LIONZ, an acronym for Lesion Segmentation, is a robust Python tool designed 
                      to act as a central platform for extracting tumor and lesion information 
                      from PET/CT datasets. Built with accuracy and performance in mind, LIONZ aims 
                      to streamline medical imaging tasks and contribute to enhanced diagnosis.''',
    keywords='lesion segmentation, tumors, PET/CT, medical imaging',
    author=authors_string,
    author_email=emails_string,
    url='https://github.com/QIMP-Team/lionz',  # Add the repo URL or documentation URL
    download_url='https://github.com/QIMP-Team/lionz/archive/v0.1.tar.gz',  # Link to the release tarball
    project_urls={
        "Bug Tracker": "https://github.com/QIMP-Team/lionz/issues",
        "Documentation": "https://lionz.readthedocs.io/en/latest/",
        "Source Code": "https://github.com/QIMP-Team/lionz",
    }
)
