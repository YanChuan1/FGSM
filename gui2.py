import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import tempfile
from urllib.request import urlretrieve
import tarfile
import os
import json
import matplotlib.pyplot as plt
import matplotlib
import PIL
import numpy as np
import cv2
from tkinter import *
from PIL import ImageTk
import sys
sys.path.append('./denoise')
import test_model

tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.InteractiveSession()
t_c = 924

def inception(image, reuse):
    preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, _ = nets.inception.inception_v3(
            preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:,1:] # ignore background class
        probs = tf.nn.softmax(logits) # probabilities
    return logits, probs


def classify(img, correct_class=None, target_class=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.sca(ax1)
    p = sess.run(probs, feed_dict={image: img})[0]
    ax1.imshow(img)
    fig.sca(ax1)
    topk = list(p.argsort()[-10:][::-1])
    topprobs = p[topk]
    barlist = ax2.bar(range(10), topprobs)
    if target_class in topk:
        barlist[topk.index(target_class)].set_color('r')
    if correct_class in topk:
        barlist[topk.index(correct_class)].set_color('g')
    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(range(10),
               [imagenet_labels[i][:15] for i in topk],
               rotation='vertical')
    fig.subplots_adjust(bottom=0.2)
    plt.show()


def start():
    l0.configure(image = img0)
    l0.image = img0
    classify(img, correct_class=img_class)



def fgsm():
    x = tf.placeholder(tf.float32, (299, 299, 3))
    x_hat = image # our trainable adversarial input
    assign_op = tf.assign(x_hat, x)
    learning_rate = tf.placeholder(tf.float32, ())
    y_hat = tf.placeholder(tf.int32, ())

    labels = tf.one_hot(y_hat, 1000)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])
    optim_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, var_list=[x_hat])

    epsilon = tf.placeholder(tf.float32, ())
    below = x - epsilon
    above = x + epsilon
    projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
    with tf.control_dependencies([projected]):
        project_step = tf.assign(x_hat, projected)

    demo_epsilon = 2.0/255.0 # a really small perturbation
    demo_lr = 1e-1
    demo_steps = 100
    demo_target = t_c # "guacamole"

    print("initialization step")
    # initialization step
    sess.run(assign_op, feed_dict={x: img})

    print("projected gradient descent")
    # projected gradient descent
    for i in range(demo_steps):
        # gradient descent step
        _, loss_value = sess.run(
            [optim_step, loss],
            feed_dict={learning_rate: demo_lr, y_hat: demo_target})
        # project step
        sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
        if (i+1) % 10 == 0:
            print('step %d, loss=%g' % (i+1, loss_value))

    adv = x_hat.eval() # retrieve the adversarial example
    matplotlib.image.imsave('after.png', adv)
    img1 = PIL.Image.open('after.png')
    img1 = img1.convert("RGB")
    img1 = ImageTk.PhotoImage(img1)
    l1.configure(image = img1)
    l1.image = img1 
    #image_output = PIL.Image.fromarray(x_hat)
    #image_output.save("adv.png")
    classify(adv, correct_class=img_class, target_class=demo_target)

def denoise():
    print("denoise")
    test_model.start()
    img2 = PIL.Image.open('./denoise/results/denoise.png')
    img2 = img2.convert("RGB")
    ig = ImageTk.PhotoImage(img2)
    l2.configure(image = ig)
    l2.image = ig
    big_dim = max(img2.width, img2.height)
    wide = img2.width > img2.height
    new_w = 299 if not wide else int(img2.width * 299 / img2.height)
    new_h = 299 if wide else int(img2.height * 299 / img2.width)
    img2 = img2.resize((new_w, new_h)).crop((0, 0, 299, 299))
    img2 = (np.asarray(img2) / 255.0).astype(np.float32)
    classify(img2, correct_class=img_class)

def compress():
    imgcom = PIL.Image.open("./after.png")
    imgcom = imgcom.convert("RGB")
    imgcom.save("./compress.jpg")
    imgcom = PIL.Image.open("./compress.jpg")
    icom = ImageTk.PhotoImage(imgcom)
    l4.configure(image = icom)
    l4.image = icom
    big_dim = max(imgcom.width, imgcom.height)
    wide = imgcom.width > imgcom.height
    new_w = 299 if not wide else int(imgcom.width * 299 / imgcom.height)
    new_h = 299 if wide else int(imgcom.height * 299 / imgcom.width)
    imgcom = imgcom.resize((new_w, new_h)).crop((0, 0, 299, 299))
    imgcom = (np.asarray(imgcom) / 255.0).astype(np.float32)
    classify(imgcom, correct_class=img_class)

def rotate():
    an = 45
    imgro = PIL.Image.open("./after.png")
    imgro = imgro.rotate(an)
    imgro = imgro.convert("RGB")
    iro = ImageTk.PhotoImage(imgro)
    l3.configure(image = iro)
    l3.image = iro
    big_dim = max(imgro.width, imgro.height)
    wide = imgro.width > imgro.height
    new_w = 299 if not wide else int(imgro.width * 299 / imgro.height)
    new_h = 299 if wide else int(imgro.height * 299 / imgro.width)
    imgro = imgro.resize((new_w, new_h)).crop((0, 0, 299, 299))
    imgro = (np.asarray(imgro) / 255.0).astype(np.float32)
    classify(imgro, correct_class=img_class)


def target():
    t_c = e.get()



image = tf.Variable(tf.zeros((299, 299, 3)))
logits, probs = inception(image, reuse=False)
data_dir = tempfile.mkdtemp()
inception_tarball = './inception_v3_2016_08_28.tar.gz'
tarfile.open(inception_tarball, 'r:gz').extractall(data_dir)


restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/')
]
saver = tf.train.Saver(restore_vars)
saver.restore(sess, os.path.join(data_dir, 'inception_v3.ckpt'))

imagenet_json = './imagenet.json'
with open(imagenet_json) as f:
    imagenet_labels = json.load(f)

img_path = './test.jpg'
img_class = 281
img = PIL.Image.open(img_path)
img = img.convert("RGB")
big_dim = max(img.width, img.height)
wide = img.width > img.height
new_w = 299 if not wide else int(img.width * 299 / img.height)
new_h = 299 if wide else int(img.height * 299 / img.width)
img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
img = (np.asarray(img) / 255.0).astype(np.float32)




window = Tk()
window.geometry('1100x1100')
window.title('FGSM')


img0 = PIL.Image.open('test.jpg')
img0 = img0.resize((299,299))
img0 = ImageTk.PhotoImage(img0)
button1 = Button(window, text = '开始', command = start)
button1.place(x = 550, y =0)
e = Entry(window)
e.place(x=350, y = 40)
bu = Button(window, text = '输入目标分类', command = target)
bu.place(x = 500 , y = 40)
button2 = Button(window, text = '生成对抗样本',command = fgsm)
button2.place(x = 400, y=80)
button3 = Button(window, text = '去噪', command = denoise)
button3.place(x = 500, y = 80)
button4 = Button(window, text = '旋转', command = rotate)
button4.place(x = 540, y = 80)
button5 = Button(window, text = '压缩', command = compress)
button5.place(x = 580, y = 80)

t0 = Label(window, text = '原图：')
t0.place(x = 0, y = 200)

l0=Label(window)
l0.place(x = 40, y = 200)#原图

t1 = Label(window, text = '对抗样本')
t1.place(x = 340 , y = 200)

l1 = Label(window)
l1.place(x = 380, y =200)#对抗样本

t2 = Label(window, text = '去噪后：')
t2.place(x = 680, y =200)

l2 = Label(window)
l2.place(x = 720, y = 200)#去噪后

t3 = Label(window, text = '旋转后：')
t3.place(x = 0, y = 510)
l3 = Label(window)
l3.place(x = 40, y = 510)#旋转后

t4 = Label(window, text = '压缩后：')
t4.place(x = 340 , y = 510)
l4 = Label(window)
l4.place(x = 380, y =510)#压缩后


window.mainloop()