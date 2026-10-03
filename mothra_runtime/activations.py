# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2007 Julien Pommier
# Copyright (C) 2014 Pedro Gonnet
# Copyright (C) 2009-2019 Gael Guennebaud
# This Source Code Form is subject to the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, obtain one at
# https://mozilla.org/MPL/2.0/.
# Adapted from Eigen headers distributed with tensorflow-macos 2.9.1.
# See LICENSE-EIGEN-NOTICE.md for attribution and source references.
"""Historical CPU float32 activations, with Keras-compatible analytic gradients.

Float64 intermediates approximate the ARM64 fused multiply-add rounding.
This preserves the tested legacy CPU path, not arbitrary historical GPU kernels.
"""
import tensorflow as tf
import keras

def multiply_add(x,y,z):
    x=tf.cast(tf.convert_to_tensor(x,dtype=tf.float32),tf.float64)
    y=tf.cast(tf.convert_to_tensor(y,dtype=tf.float32),tf.float64)
    z=tf.cast(tf.convert_to_tensor(z,dtype=tf.float32),tf.float64)
    return tf.cast(x*y+z,tf.float32)

@keras.saving.register_keras_serializable(package="Mothra")
@tf.custom_gradient
def legacy_tanh(a):
    a=tf.convert_to_tensor(a,dtype=tf.float32)
    x=tf.clip_by_value(a,-7.90531110763549805,7.90531110763549805)
    x2=x*x
    p=multiply_add(x2,-2.76076847742355e-16,2.00018790482477e-13)
    for coefficient in [-8.60467152213735e-11,5.12229709037114e-08,1.48572235717979e-05,6.37261928875436e-04,4.89352455891786e-03]:
        p=multiply_add(x2,p,coefficient)
    p=x*p
    q=multiply_add(x2,1.19825839466702e-06,1.18534705686654e-04)
    q=multiply_add(x2,q,2.26843463243900e-03)
    q=multiply_add(x2,q,4.89352518554385e-03)
    y=tf.where(tf.abs(a)<0.0004,x,p/q)
    return y,lambda grad:grad*(1-y*y)

@keras.saving.register_keras_serializable(package="Mothra")
@tf.custom_gradient
def legacy_sigmoid(a):
    a=tf.convert_to_tensor(a,dtype=tf.float32)
    x=tf.minimum(a,88.723)
    m=tf.floor(multiply_add(x,1.44269504088896341,0.5))
    r=multiply_add(m,-0.693359375,x)
    r=multiply_add(m,2.12194440e-4,r)
    r2=r*r
    even=multiply_add(r2,1.37449637986719608306884765625e-3,4.166965186595916748046875e-2)
    odd=multiply_add(r2,8.36894474923610687255859375e-3,0.16666518151760101318359375)
    even=multiply_add(r2,even,0.49999988079071044921875)
    p=multiply_add(r,odd,even)
    p=multiply_add(r2,p,r+1)
    exponent=tf.cast(tf.clip_by_value(m,-278.,278.),tf.int32)
    quarter=tf.bitwise.right_shift(exponent,2)
    def power_of_two(value):
        return tf.bitcast(tf.bitwise.left_shift(value+127,23),tf.float32)
    scale=power_of_two(quarter)
    e=((p*scale)*scale)*scale
    e=tf.maximum(e*power_of_two(exponent-3*quarter),a)
    e=tf.where(a < -104.,0.,e)
    y=tf.where(tf.math.is_inf(e),1.,e/(1+e))
    return y,lambda grad:grad*y*(1-y)

