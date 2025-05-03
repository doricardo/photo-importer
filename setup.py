from setuptools import setup

APP = ['photo_importer_app.py']
DATA_FILES = [
    ('models', [
        'models/opencv_face_detector_uint8.pb',
        'models/opencv_face_detector.pbtxt'
    ]),
    'icon.png'
]
OPTIONS = {
    'argv_emulation': True,
    'includes': ['cv2','PIL','PySide6'],
    'iconfile': 'icon.png',
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
