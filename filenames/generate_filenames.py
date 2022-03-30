import os
from glob import glob


def gen_kitti_2015():
    data_dir = 'data/KITTI/kitti_2015/data_scene_flow'

    train_file = 'KITTI_2015_train.txt'
    val_file = 'KITTI_2015_val.txt'

    # Split the training set with 4:1 raito (160 for training, 40 for validation)
    with open(train_file, 'w') as train_f, open(val_file, 'w') as val_f:
        dir_name = 'image_2'
        left_dir = os.path.join(data_dir, 'training', dir_name)
        left_imgs = sorted(glob(left_dir + '/*_10.png'))

        print('Number of images: %d' % len(left_imgs))

        for left_img in left_imgs:
            right_img = left_img.replace(dir_name, 'image_3')
            disp_path = left_img.replace(dir_name, 'disp_occ_0')

            img_id = int(os.path.basename(left_img).split('_')[0])

            if img_id % 5 == 0:
                val_f.write(left_img.replace(data_dir + '/', '') + ' ')
                val_f.write(right_img.replace(data_dir + '/', '') + ' ')
                val_f.write(disp_path.replace(data_dir + '/', '') + '\n')
            else:
                train_f.write(left_img.replace(data_dir + '/', '') + ' ')
                train_f.write(right_img.replace(data_dir + '/', '') + ' ')
                train_f.write(disp_path.replace(data_dir + '/', '') + '\n')

def gen_apolloscape():
    data_dir = 'data/apolloscape'

    train_file = 'apolloscape_train.txt'
    val_file = 'apolloscape_val.txt'

    ## Generate training lists
    train_lists = []
    # train_dir = os.path.join(data_dir, "/stereo_train")
    train_dir = data_dir + "/stereo_train"
    for train_img in os.listdir(train_dir + "/disparity"):
        train_lists.append(train_img)

    with open(train_file, 'w') as train_f:
        for img_name in train_lists:
            # left_img = train_dir + '/camera_5/' + img_name[0: len(img_name) - 5] + '.jpg'
            # right_img = train_dir + '/camera_6/' + img_name[0: len(img_name) - 6] + '6.jpg'
            # disp_img = train_dir + '/disparity/' + img_name
            left_img = 'stereo_train/camera_5/' + img_name[0: len(img_name) - 5] + '5.jpg'
            right_img = 'stereo_train/camera_6/' + img_name[0: len(img_name) - 5] + '6.jpg'
            disp_img = 'stereo_train/disparity/' + img_name

            train_f.write(left_img + ' ')
            train_f.write(right_img + ' ')
            train_f.write(disp_img + '\n')
    train_f.close()

    ## Generate validation lists
    val_lists = []
    # val_dir = os.path.join(data_dir, "/stereo_test")
    val_dir = data_dir + "/stereo_test"
    for val_img in os.listdir(val_dir + "/disparity"):
        val_lists.append(val_img)

    with open(val_file, 'w') as val_f:
        for img_name in val_lists:
            left_img = 'stereo_test/camera_5/' + img_name[0: len(img_name) - 5] + '5.jpg'
            right_img = 'stereo_test/camera_6/' + img_name[0: len(img_name) - 5] + '6.jpg'
            disp_img = 'stereo_test/disparity/' + img_name

            val_f.write(left_img + ' ')
            val_f.write(right_img + ' ')
            val_f.write(disp_img + '\n')


def gen_apolloscape_road():
    data_dir = 'data/apolloscape'

    # train_file = 'apolloscape_train.txt'
    val_file = 'apolloscape_val.txt'

    ## Generate training lists
    # train_lists = []
    # # train_dir = os.path.join(data_dir, "/stereo_train")
    # train_dir = data_dir + "/stereo_train"
    # for train_img in os.listdir(train_dir + "/disparity"):
    #     train_lists.append(train_img)

    # with open(train_file, 'w') as train_f:
    #     for img_name in train_lists:
    #         # left_img = train_dir + '/camera_5/' + img_name[0: len(img_name) - 5] + '.jpg'
    #         # right_img = train_dir + '/camera_6/' + img_name[0: len(img_name) - 6] + '6.jpg'
    #         # disp_img = train_dir + '/disparity/' + img_name
    #         left_img = 'stereo_train/camera_5/' + img_name[0: len(img_name) - 5] + '5.jpg'
    #         right_img = 'stereo_train/camera_6/' + img_name[0: len(img_name) - 5] + '6.jpg'
    #         disp_img = 'stereo_train/disparity/' + img_name

    #         train_f.write(left_img + ' ')
    #         train_f.write(right_img + ' ')
    #         train_f.write(disp_img + '\n')
    # train_f.close()

    ## Generate validation lists
    val_lists = []
    # val_dir = os.path.join(data_dir, "/stereo_test")
    val_dir = data_dir + "/road02_seg/Depth/Record022/Camera 5"
    for val_img in os.listdir(val_dir):
        val_lists.append(val_img)

    with open(val_file, 'w') as val_f:
        for img_name in val_lists:
            left_img = 'road02_seg/ColorImage/Record022/Camera 5/' + img_name[0: len(img_name) - 5] + '5.jpg'
            right_img = 'road02_seg/ColorImage/Record022/Camera 6/' + img_name[0: len(img_name) - 5] + '6.jpg'
            disp_img = "road02_seg/Depth/Record022/Camera 5" + img_name

            val_f.write(left_img + ' ')
            val_f.write(right_img + ' ')
            val_f.write(disp_img + '\n')


if __name__ == '__main__':
    # gen_kitti_2015()
    # gen_apolloscape()
    gen_apolloscape_road()
