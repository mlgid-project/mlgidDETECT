import numpy as np
from h5py import File
from pathlib import Path
from mlgiddetect.configuration import Config
from mlgiddetect.dataloader import pygid_results_dtype, get_results_array

def export_pygid_h5(config: Config, img_container):

    if img_container.config.INPUT_IMGPATH:
        filename = Path(img_container.config.OUTPUT_FOLDER) / (Path(img_container.config.INPUT_IMGPATH).stem + '.h5')
    else:
        filename = Path(img_container.config.OUTPUT_FOLDER) / "detection_results.h5"

    #append a numbering if file already exists:
    filename = next(filename.with_name(f"{filename.stem}_{i}{filename.suffix}") for i in range(1000) if not filename.with_name(f"{filename.stem}_{i}{filename.suffix}").exists())

    with File(filename, 'a') as f:
        source_path = f'{filename.stem}/data/analysis/frame' +  str(img_container.nr).zfill(5) +'/'
        data_group = f.create_group(f'{filename.stem}/data/')
        data_group.create_dataset('img_gid_q', data=img_container.raw_reciprocal[np.newaxis,:,:,])
        
        height, width = img_container.raw_reciprocal.shape
        q_z_value = img_container.q_z if img_container.q_z is not None else 2.7
        data_group.create_dataset('q_z', data=np.linspace(0, q_z_value, width))
        q_xy_value = img_container.q_xy if img_container.q_xy is not None else 2.7
        data_group.create_dataset('q_xy', data=np.linspace(0, q_xy_value, height))

        group = f.create_group(source_path)
        results_struct = get_results_array(img_container)
        group.create_dataset('detected_peaks', data=results_struct, dtype=pygid_results_dtype)