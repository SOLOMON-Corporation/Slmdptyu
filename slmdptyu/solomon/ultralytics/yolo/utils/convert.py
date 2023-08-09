import numpy as np

xv, yv = np.meshgrid(np.arange(8), np.arange(9))
discrete_angles = ((xv*45 + yv*5) * np.pi / 180).flatten()

def weightedMeanOnCircle(weights):
    x = np.sum(np.cos(discrete_angles) * weights) / len(discrete_angles)
    y = np.sum(np.sin(discrete_angles) * weights) / len(discrete_angles)
    return np.arctan2(y, x)

def MeanShiftCircle(weights_list):
    results = []
    kappa = 5
    for weights in weights_list:
        basePoint = discrete_angles[0]
        initialPoint = discrete_angles[0]
        numPoints = len(discrete_angles)
        flags = np.zeros(numPoints)
        maxMode = 0
        modeAmp = 0
        while True:
            preBasePoint = basePoint
            dis = np.arccos(np.cos(discrete_angles - basePoint))+1e-15
            temp = np.where(dis != 0, np.sin(dis) / dis, 1)
            nnWeights = weights * kappa / 2 * temp * np.exp(kappa * np.cos(dis))
            basePoint = weightedMeanOnCircle(nnWeights)

            if np.cos(preBasePoint - basePoint) >= np.cos(0.01 * np.pi / 180):
                middlePoint = np.arctan2(np.sin(basePoint) + np.sin(initialPoint),
                                        np.cos(basePoint) + np.cos(initialPoint))
                flags = np.where(np.cos(discrete_angles - middlePoint) >= np.cos(basePoint - middlePoint), 1, flags)
                # amp = 0
                amp = np.sum(weights * np.exp(kappa * np.cos(discrete_angles - basePoint)))
                if modeAmp <= amp:
                    modeAmp = amp
                    maxMode = basePoint
                nextInitialPoint = 0
                flg = 0
                for i in range(len(flags)):
                    if flags[i] == 0:
                        if flg == 0:
                            nextInitialPoint = discrete_angles[i]
                            flags[i] = 1
                            flg = 1
                        else:
                            if discrete_angles[i] == nextInitialPoint:
                                flags[i] = 1
                if flg == 0:
                    break
                basePoint = nextInitialPoint
                initialPoint = nextInitialPoint
        if (maxMode < 0):
            maxMode += (2 * np.pi)
        results.append(maxMode)
    return results

def hms_2_points(pred_heatmaps, pred_masks):
    np.seterr(divide='ignore', invalid='ignore')
    num_objects, num_keypoints, h, w = pred_heatmaps.shape
    pred_masks = np.tile(pred_masks.astype(np.bool_)[:, None, ...], [1, num_keypoints, 1, 1])
    pred_heatmaps_tmp = np.zeros_like(pred_heatmaps, dtype=np.uint8)
    pred_heatmaps_tmp[pred_masks] = pred_heatmaps[pred_masks]

    meshgrid = np.meshgrid(range(w), range(h))

    locations = np.where(
        np.tile(
            (pred_heatmaps_tmp == \
                np.tile(
                    np.max(pred_heatmaps_tmp, axis=(2, 3))[..., None, None], 
                    (1, 1, h, w)))[..., None],
            [1, 1, 1, 1, 2]
            ),
        np.tile(np.concatenate(
            [i[..., None].astype(np.int16) for i in meshgrid], 
            axis=-1)[None, ...], [num_objects, num_keypoints, 1, 1, 1]),
        np.zeros((num_objects, num_keypoints, h, w, 2), dtype=np.int16)
    )
    kpts = locations.sum(axis=(2, 3))/(locations!=0).sum(axis=(2, 3))

    kpts += 0.5 # center of pixel

    filter_nan = np.tile(np.isnan(kpts).any(axis=-1)[..., None], [1, 1, 2])
    kpts = np.where(filter_nan, 0, kpts)

    return kpts