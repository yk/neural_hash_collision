#!/usr/bin/env python3

#partially based on https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX/blob/master/nnhash.py

import tensorflow as tf
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks import evasion
import numpy as np
from PIL import Image

seed_fn = './neuralhash_128x96_seed1.dat'

seed1 = open(seed_fn, 'rb').read()[128:]
seed1 = np.frombuffer(seed1, dtype=np.float32)
seed1 = seed1.reshape([96, 128])

def _neural_hash(out):
    hash_output = seed1.dot(out.numpy().flatten())
    hash_bits = ''.join(['1' if it >= 0 else '0' for it in hash_output])
    hash_hex = '{:0{}x}'.format(int(hash_bits, 2), len(hash_bits) // 4)
    return hash_hex

def _load_img(fname):
    img = Image.open(fname)
    img = img.convert('RGB')
    img = img.resize([360, 360])
    img = np.array(img).astype(np.float32) / 255.0
    img = img * 2.0 - 1.0
    img = img.transpose(2, 0, 1).reshape([1, 3, 360, 360])
    return img

model = tf.saved_model.load('./model.pb')
last_input = None
def _model(inp, **kwargs):
    global last_input
    last_input = np.copy(inp)
    output = model(image=inp)[0][0, :, 0, 0]
    return output

source_img = _load_img('./doge.jpeg')
target_img = _load_img('./titanic.jpeg')
loss_object = tf.keras.losses.CosineSimilarity()

target_hash = _model(target_img)
target_neural_hash = _neural_hash(target_hash)

print('TARGET HASH: ', target_neural_hash)

def _save_img(img, fname):
    img = img.transpose(1, 2, 0)
    img = img + 1.
    img = img / 2.
    img = img.clip(0., 1.)
    img = np.uint8(img * 255) # final image might not have the exact same hash due to clipping & quantization
    img = Image.fromarray(img, 'RGB')
    img.save(fname, quality=100)

def _loss(dummy, img_hash, *args, **kwargs):
    loss = loss_object(img_hash, target_hash)
    print('Loss: ', loss.numpy().item())
    neural_hash = _neural_hash(img_hash)
    print('HASH: ', neural_hash)
    if neural_hash == target_neural_hash:
        print('-------------')
        print('Collision found!!!')
        print('-------------')
        _save_img(last_input[0], 'collision.jpeg')
        exit(0) # we could also continue here & decrease the step size to lower the loss even more
    return -loss

classifier = TensorFlowV2Classifier(
    model=_model,
    loss_object=_loss,
    nb_classes=128,
    input_shape=(3, 360, 360),
    clip_values=(-1, 1),
)

attack = evasion.ProjectedGradientDescentTensorFlowV2(
        estimator=classifier, 
        norm=2, # other common option is np.inf here, but eps and eps_step need to be changed
        eps=60., # if no success, raise this. for less artifacts, lower this.
        eps_step=.5, # raising this makes it go faster, but more noisy
        targeted=False,
        max_iter=5000,
        )

result = attack.generate(x=source_img, y=(0,))

