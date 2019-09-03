import tensorflow as tf
import matplotlib.image as img
import matplotlib.pyplot as plot

myfile = "../../resource/plot/img8.jpg"
myimage = img.imread(myfile)
print(myimage.ndim)  # 3
print(myimage.shape)  # (1200, 1920, 3)  高为1200、宽为1920、颜色深度为3(即RGB的3个值)
plot.imshow(myimage)
plot.show()  # 手工保存打印出来的图片会减少像素
plot.imsave('../../output/plot/img8_111.jpg', myimage)  # 用这个接口保存像素不减少

myslice = tf.placeholder("int32", [None, None, 3])
cropped = tf.slice(myimage, [0, 300, 0], [-1, -1, -1])
# 高从0到结束，宽从300到结束，深度从0到结束，所以就是把宽0~300部分裁掉了
sess = tf.Session()
result = sess.run(cropped, feed_dict={myslice: myimage})
plot.imshow(result)
plot.show()
plot.imsave('../../output/plot/img8_222.jpg', result)

image1 = tf.Variable(myimage, name='image')
myvars = tf.global_variables_initializer()
# sess = tf.Session()
flipped = tf.transpose(image1, perm=[1, 0, 2])
# 用transpose翻转输入网格的 0 轴和 1 轴，原本是[0, 1, 2]这样3维
sess.run(myvars)
result1 = sess.run(flipped)
print(flipped.shape)
print(result1.shape)
plot.imshow(result1)
plot.show()
plot.imsave('../../output/plot/img8_333.jpg', result1)

image2 = tf.image.rot90(myimage, k=1)
# 用image.rot90旋转图片，k是旋转90度的次数
result2 = sess.run(image2)
plot.imshow(result2)
plot.show()
plot.imsave('../../output/plot/img8_444.jpg', result2)

image3 = tf.image.rot90(myimage, k=2)
# 用image.rot90旋转图片，k是旋转90度的次数
result3 = sess.run(image3)
print(image3.shape)  # (?, ?, 3)
print(result3.shape)  # (1200, 1920, 3)
plot.imshow(result3)
plot.show()
