import numpy as np
import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
# tf.compat.v1.disable_eager_execution()
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
import matplotlib.cm as cm
# import cv2
from tensorflow.keras import backend as K

# https://keras.io/examples/vision/grad_cam/
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_gradcam(img_path, heatmap, image_size, cam_path, alpha=0.4):
    # Load the original image
    img = load_img(
        img_path,
        color_mode='rgb'
    )
    img = tf.image.resize(img, [image_size, image_size], preserve_aspect_ratio=True)
    img = img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)
    return cam_path

# def grad_cam(input_model, image, cls, layer_name, image_size):
#     """GradCAM method for visualizing input saliency."""
#     y_c = input_model.output[0, cls]
#     conv_output = input_model.get_layer(layer_name).output
#     grads = K.gradients(y_c, conv_output)[0]
#     #normalize if necessary
#     gradient_function = K.function([input_model.input], [conv_output, grads])

#     output, grads_val = gradient_function([image])
#     output, grads_val = output[0, :], grads_val[0, :, :, :]

#     weights = np.mean(grads_val, axis=(0, 1))
#     cam = np.dot(output, weights)
#     # print(cam)

#     #process CAM
#     cam = cv2.resize(cam, (image_size, image_size), cv2.INTER_LINEAR)
#     cam = np.maximum(cam, 0)
#     cam = cam / cam.max()
#     return cam

# def compute_saliency(model, img_path, image_size, cam_path, layer_name='top_activation', cls=-1):
#     """Compute saliency using all three approaches.
#         -layer_name: layer to compute gradients;
#         -cls: class number to localize (-1 for most probable class).
#     """

#     img = load_img(
#         img_path,
#         color_mode='rgb',
#         target_size=(image_size, image_size)
#     )
#     preprocessed_input = np.expand_dims(img_to_array(img)/255, axis=0)
# #     print(preprocessed_input)

#     predictions = model.predict(preprocessed_input)
#     # print(predictions)
# #     top = decode_predictions(predictions, top=top_n)[0]
# #     classes = np.argsort(predictions[0])[-top_n:][::-1]
# #     print('Model prediction:')
# #     for c, p in zip(classes, top):
# #         print('\t{:15s}\t({})\twith probability {:.3f}'.format(p[0], c, p[1]))
#     if cls == -1:
#         cls = np.argmax(predictions)
#     # print(cls)
# #     class_name = decode_predictions(np.eye(1, 2, cls))[0][0][0]
# #     print("Explanation for '{}'".format(class_name))
    
#     gradcam = grad_cam(model, preprocessed_input, cls, layer_name, image_size)
#     # print(gradcam)

#     jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
#     # print(img)
#     jetcam = (np.float32(jetcam) + img) / 2
#     cv2.imwrite(cam_path, np.uint8(jetcam))
        
#     return predictions[0][1]