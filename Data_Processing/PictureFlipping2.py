import tensorflow as tf
import matplotlib.pyplot as plt
import imageio as imgio
from skimage import transform

myfile = "../../resource/plot/img8.jpg"

miscimg = imgio.imread(myfile)
angle = 90.0
image2 = transform.rotate(miscimg, angle, resize=True)
# 这里如果不加resize=True的话，会维持旋转前的高和宽，图片内容就少了
plt.imshow(image2)
plt.show()


img = tf.gfile.FastGFile(myfile, 'rb').read()  # 这里不加rb会报错
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(img)
    # 将图像上下翻转
    flipped0 = tf.image.flip_up_down(img_data)
    # 将图像左右翻转
    flipped1 = tf.image.flip_left_right(img_data)
    # 通过交换第一维和第二维来转置图像
    flipped2 = tf.image.transpose_image(img_data)

    plt.subplot(221), plt.imshow(img_data.eval()), plt.title('original')
    plt.subplot(222), plt.imshow(flipped0.eval()), plt.title('flip_up_down')
    plt.subplot(223), plt.imshow(flipped1.eval()), plt.title('flip_left_right')
    plt.subplot(224), plt.imshow(flipped2.eval()), plt.title('transpose_image')

    plt.show()

    result0 = sess.run(img_data)
    result1 = sess.run(flipped0)
    result2 = sess.run(flipped1)
    result3 = sess.run(flipped2)
    print(result0.shape)
    print(result1.shape)
    print(result2.shape)
    print(result3.shape)

    plt.imsave('../../output/plot/img8_555.jpg', result1)
    plt.imsave('../../output/plot/img8_666.jpg', result2)
    plt.imsave('../../output/plot/img8_777.jpg', result3)
