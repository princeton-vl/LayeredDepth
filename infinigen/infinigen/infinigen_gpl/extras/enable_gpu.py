# This code is derived from https://github.com/DLR-RM/BlenderProc/blob/main/blenderproc/python/utility/Initializer.py under a GPL-3.0 license. 

import bpy
import logging

logger = logging.getLogger(__name__)

def enable_gpu(engine_name = 'CYCLES'):
    compute_device_type = None
    prefs = bpy.context.preferences.addons['cycles'].preferences
    # Use cycles
    bpy.context.scene.render.engine = engine_name
    bpy.context.scene.cycles.device = 'GPU'

    preferences = bpy.context.preferences.addons['cycles'].preferences
    for device_type in preferences.get_device_types(bpy.context):
        preferences.get_devices_for_type(device_type[0])

    for gpu_type in ['OPTIX', 'CUDA']:#, 'METAL']:
        found = False
        for device in preferences.devices:
            if device.type == gpu_type and (compute_device_type is None or compute_device_type == gpu_type):
                bpy.context.preferences.addons['cycles'].preferences.compute_device_type = gpu_type
                logger.info('Device {} of type {} found and used.'.format(device.name, device.type))
                found = True
                break
        if found:
            break

    # make sure that all visible GPUs are used
    for device in prefs.devices:
        device.use = True
    return prefs.devices
