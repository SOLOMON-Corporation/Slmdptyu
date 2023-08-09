import tensorflow as tf


class TF_model_w_PP:
    def __init__(
            self,
            model,
            nc=80,
            nk=17,
            w_mask=False,
            w_keypoints=False,
            w_angle=False,
            topk=100,
            iou_thr=0.5,
            conf_thr=0.25
    ):
        self.model = tf.saved_model.load(model)
        self.nc = nc
        self.nk = nk
        self.w_mask = w_mask
        self.w_keypoints = w_keypoints
        self.w_angle = w_angle
        self.topk = topk
        self.iou_thr = iou_thr
        self.conf_thr = conf_thr

    @tf.function
    def yolov8_w_pp(self, img):
        prediction = self.model(img)
        if type(prediction) == list:
            p, proto = prediction
            _, mh, mw, _ = proto.shape
            _, ih, iw, _ = img.shape
        else:
            p = prediction

        x = tf.transpose(p[0], perm=[1, 0])
        if self.w_keypoints and self.w_mask and self.w_angle:
            box, cls, ang, mask, key = tf.split(x, (4, self.nc, 72, 32, self.nk * 3), axis=1)
        elif self.w_keypoints and self.w_mask:
            box, cls, mask, key = tf.split(x, (4, self.nc, 32, self.nk * 3), axis=1)
        elif self.w_keypoints and self.w_angle:
            box, cls, ang, key = tf.split(x, (4, self.nc, 72, self.nk * 3), axis=1)
        elif self.w_keypoints:
            box, cls, key = tf.split(x, (4, self.nc, self.nk * 3), axis=1)
        elif self.w_mask and self.w_angle:
            box, cls, ang, mask = tf.split(x, (4, self.nc, 72, 32), axis=1)
        elif self.w_mask:
            box, cls, mask = tf.split(x, (4, self.nc, 32), axis=1)
        elif self.w_angle:
            box, cls, ang = tf.split(x, (4, self.nc, 72), axis=1)
        else:
            box, cls = tf.split(x, (4, self.nc), axis=1)
        conf = tf.math.reduce_max(cls, axis=1)
        cls = tf.math.argmax(cls, axis=1)

        # xywh2xyxy
        boxes = tf.concat((
            (box[..., 1] - box[..., 3] / 2)[..., None],  # y1
            (box[..., 0] - box[..., 2] / 2)[..., None],  # x1
            (box[..., 1] + box[..., 3] / 2)[..., None],  # y2
            (box[..., 0] + box[..., 2] / 2)[..., None]),  # x2
            axis=1
        )

        # NMS (agnostic)
        selected_indices, scores, valid = tf.raw_ops.NonMaxSuppressionV5(
            boxes=boxes,
            scores=conf,
            max_output_size=self.topk,
            iou_threshold=self.iou_thr,
            score_threshold=self.conf_thr,
            soft_nms_sigma=0
        )
        boxes = tf.gather_nd(boxes, selected_indices[..., None])
        clsids = tf.gather_nd(cls, selected_indices[..., None])
        clsids = tf.cast(clsids, dtype=tf.int32)

        boxes = tf.gather(boxes, [1, 0, 3, 2], axis=-1)  # yxyx to xyxy

        results = [valid, boxes, scores, clsids]

        if self.w_mask:
            mask = tf.gather_nd(mask, selected_indices[..., None])
            masks = tf.einsum("km,ijm->kij", mask, proto[0])
            masks = tf.math.sigmoid(masks) > 0.5

            w_ratio = mw / iw
            h_ratio = mh / ih
            x1 = (boxes[..., 0] * w_ratio)[:, None, None]
            y1 = (boxes[..., 1] * h_ratio)[:, None, None]
            x2 = (boxes[..., 2] * w_ratio)[:, None, None]
            y2 = (boxes[..., 3] * h_ratio)[:, None, None]

            r = tf.range(mw, dtype=tf.float32)[None, None, :]
            c = tf.range(mh, dtype=tf.float32)[None, :, None]
            r_valid = tf.math.logical_and((r >= x1), (r < x2))
            c_valid = tf.math.logical_and((c >= y1), (c < y2))
            m_valid = tf.math.logical_and(r_valid, c_valid)
            masks = tf.where(
                m_valid,
                tf.cast(masks, dtype=tf.bool),
                tf.constant(False, dtype=tf.bool)
            )
            results.append(masks)

        if self.w_keypoints:
            key = tf.gather_nd(key, selected_indices[..., None])
            results.append(key)

        if self.w_angle:
            angles = tf.gather_nd(ang, selected_indices[..., None])
            results.append(angles)

        return results

    def export(self, filename):
        concrete_func = self.yolov8_w_pp.get_concrete_function(
            *[
                tf.TensorSpec(
                    self.model.inputs[0].shape,
                    self.model.inputs[0].dtype
                ),  # img
            ]
        )

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        tflite_model = converter.convert()

        with open(filename, 'wb') as f:
            f.write(tflite_model)
