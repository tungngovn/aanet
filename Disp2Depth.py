import numpy as np
from PIL import Image, ImageDraw
from numpy import asarray
from numpy.core.fromnumeric import shape
# import open3d as o3d
import matplotlib.pyplot as plt
import json
import scipy.stats
import math

class Disp2Depth:

    def __init__(self, intrinsic, baseline):
        pass
        self.intrinsic = intrinsic
        self.baseline = baseline
        self.threshold = 100 # distance threshold

    def disp2depth(self, disp_image):
        # ratio = 0.3987220447284345
        ratio = 1
        focal_x = self.intrinsic[0,0]*ratio
        ## KITTI specs
        # focal_x = 721
        # baseline = 0.54
        baseline = self.baseline

        # print('Focal x: ', focal_x)
        # print('baseline: ', baseline)

        depth_image = np.zeros(disp_image.shape)
        depth_image = depth_image + (focal_x * baseline)
        # disp_image = disp_image/256
        depth_image = np.divide(depth_image, disp_image)

        return depth_image
    
    def plotRelErr(true_depth_img, depth_err_img, graph_name):
        pass
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Depth error related to groundtruth depth')
        ax1.scatter(true_depth_img, depth_err_img, s=1)
        # ax1.scatter(true_depth_img[:,:], depth_err_img[:,:], s=1)
        ax1.set_xlabel('Groundtruth depth (m)')
        ax1.set_ylabel('Depth error (m)')
        ax1.set_title('Depth error related to depth')
        ax2.scatter(true_depth_img, (depth_err_img/true_depth_img)*100, s=1)
        # ax2.scatter(true_depth_img[:,:], (depth_err_img[:,:]/true_depth_img[:,:])*100, s=1)
        ax2.set_xlabel('Predicted depth (m)')
        ax2.set_ylabel('Depth error (%)')
        ax2.set_title('Depth error related to depth')
        fig.show()
        fig.savefig(graph_name)


    def crop_gt(gt_img, crop_height, crop_width):
        pass
        # Crop gt image at its center
        h, w = np.shape(gt_img)
        start_x = (w-crop_width)//2
        start_y = (h-crop_height)//2
        cropped_img = gt_img[start_y:start_y+crop_height, start_x:start_x+crop_width]
        return cropped_img

    def save_depth_img(image, name):
        pass

        mask_pos = (image > 255)
        mask_neg = (image < -255)

        int_image = np.uint8(image)
        int_image[mask_pos] = 255
        int_image[mask_neg] = 255
        depth_image = Image.fromarray(int_image)
        depth_image.save(name)

    def image2text(image, name):
        file = open(name, "w")
        shape_image = np.shape(image)
        for i in range(shape_image[0]):
            for j in range(shape_image[1]):
                file.write('%.3f' %image[i,j])
                file.write(' ')
            file.write('\n')
        file.close()

    def readBbox(annotation_file, img_name, img_width, img_height):
        # img_path = './disparity/180116_064235315_Camera_5.png'
        # img = Image.open(img_path)
        f = open(annotation_file)
        bbox_list = []
        while True:
            line = f.readline()
            if line == '':
                print('End of annotation file')
                break
            line = line.replace("\n","")
            c, x_center, y_center, w, h = line.split(' ')

            c = int(c)
            x_center = int(float(x_center)*img_width)
            y_center = int((float(y_center))*img_height)
            w = int(float(w)*img_width)
            h = int(float(h)*img_height)

            # PIL coordinates are different from OpenCV coordinates
            x_min = int(x_center - w/2)
            y_min = int(y_center - h/2)

            bbox = [c, x_min, y_min, x_min + w, y_min + h]
            shape = [x_min, y_min, x_min + w, y_min + h]

            bbox_list.append(bbox)

            # drawImg = ImageDraw.Draw(img)
            # drawImg.rectangle(shape, fill='red', outline='blue')
            # # color_img.show()
            # img.save('out.png')

        return bbox_list
        

    def readBboxResult(self, result_json_file, img_name, img_width, img_height):
        f = open(result_json_file)
        data = json.load(f)
        img = [img for img in data if img_name in img['filename']]
        print(img[0]['filename'])
        print(img)
        bbox_list = []
        for obj in img[0]['objects']:
            if obj['class_id'] == 0: c = 82 
            else: c = 83
            center_x = obj['relative_coordinates']['center_x']
            center_y = obj['relative_coordinates']['center_y']
            w_obj = obj['relative_coordinates']['width']
            h_obj = obj['relative_coordinates']['height']
            print('center x: ', center_x)

            x_center = int(float(center_x)*img_width)
            y_center = int((float(center_y))*img_height)
            w = int(float(w_obj)*img_width)
            h = int(float(h_obj)*img_height)

            # PIL coordinates are different from OpenCV coordinates
            x_min = int(x_center - w/2)
            y_min = int(y_center - h/2)

            bbox = [c, x_min, y_min, x_min + w, y_min + h]

            bbox_list.append(bbox)

        return bbox_list
        # pass


    def drawPCL(self, disp_image, PCL_name):
        focal_x = self.intrinsic[0,0]
        baseline = self.baseline
        # disp_img = 'disparity/171206_034625454_Camera_5.png'

        # disp_image = asarray(Image.open(disp_img))
        h,w = disp_image.shape[:2]
        # h,w = depth_image.shape[:2]
        print('h: ', h)
        print('w: ', w)


        print('Focal x: ', focal_x)
        # print('baseline: ', baseline)
        disp_image = disp_image/200
        depth_image = np.ones(disp_image.shape) * focal_x * baseline
        
        depth_image = np.divide(depth_image, disp_image)
        # print(disp_image.shape)
        # print(depth_image.shape)

        # Eliminate pixels: depth > threshold
        a = np.argwhere(depth_image > self.threshold)

        for i in range(len(a)):
            depth_image[a[i,0], a[i,1]] = 0
            disp_image[a[i,0], a[i,1]]  = 100
            pass


        xdata = np.zeros((w*h,1))
        ydata = np.zeros((w*h,1))
        zdata = np.zeros((w*h,1))
        # print(disp_image.shape)

        print(xdata.shape)

        for j in range(h):
            xdata[j*w:j*w+w,0] = np.divide((np.arange(w)-w/2) * baseline, disp_image[j])
            ydata[j*w:j*w+w,0] = -np.divide((np.ones(w)*j-h/2) * baseline, disp_image[j])
            zdata[j*w:j*w+w,0] = -depth_image[j]


        xyz = np.concatenate((xdata,ydata,zdata), axis=1)
        # print(xyz)

        # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        o3d.io.write_point_cloud(PCL_name, pcd)

        # Load saved point cloud and visualize it
        pcd_load = o3d.io.read_point_cloud(PCL_name)
        o3d.visualization.draw_geometries([pcd_load])

class SensorModel:
    def __init__(self, true_depth, pred_depth) -> None:
        self.pred_depth = pred_depth
        self.true_depth = true_depth
        self.sigma = 1
        self.z_max = 100
        pass

    ## Gaussian distribution
    def p_hit(self, z, z_exp, sigma):
        phit = scipy.stats.norm.pdf(z, z_exp, math.sqrt(sigma))
        return phit
        # pass
    
    ## Maximum sensor measurements
    def p_max(self, z, z_max):
        if z == z_max: return 1
        else: return 0
        # pass

    ## Random measurements
    def p_rand(self, z, z_max):
        if z < z_max: return 1/z_max
        else: return 0
        # pass

    def sensorModel(self):

        pred_depth = self.pred_depth
        true_depth = self.true_depth

        print('Prediction depth shape: ', pred_depth.shape[0])

        for k in range(10):

            e_hit = np.array([0])
            e_max = np.array([0])
            e_rand = np.array([0])

            for i in range(pred_depth.shape[0]):
                for j in range(pred_depth.shape[1]):

                    print('i: ', i)
                    print('j: ', j)

                    print('p_hit: ', self.p_hit(pred_depth[i,j], true_depth[i,j], self.sigma))
                    print('p_max: ', self.p_max(pred_depth[i,j], self.z_max))
                    print('p_rand: ', self.p_rand(pred_depth[i,j], self.z_max))

                    ng_inv = self.p_hit(pred_depth[i,j], true_depth[i,j], self.sigma) + self.p_max(pred_depth[i,j], self.z_max) + self.p_rand(pred_depth[i,j], self.z_max)
                    ng = 1/ng_inv
                    print(ng)

                    ## Calculate z_i*
                    z_i_star = true_depth[i,j]
                    print('z_i_star: ', z_i_star)

                    e_hit = np.append(e_hit, ng*self.p_hit(pred_depth[i,j], true_depth[i,j], self.sigma))
                    e_max = np.append(e_max, ng*self.p_max(pred_depth[i,j], self.z_max))
                    e_rand = np.append(e_rand, ng*self.p_rand(pred_depth[i,j], self.z_max))

            al_hit = (1/np.sum(pred_depth)) * np.sum(e_hit)
            al_max = (1/np.sum(pred_depth)) * np.sum(e_max)
            al_rand = (1/np.sum(pred_depth)) * np.sum(e_rand)

            sigma = math.sqrt(np.sum(e_hit[1:]*((pred_depth - z_i_star)**2).flatten())/(np.sum(e_hit)))

            print('Check alpha: ', (al_hit+al_max+al_rand))
            # print(al_hit)
            # print(p_hit(pred_depth[i,j], true_depth[i,j], sigma))

            p = np.copy(pred_depth)
            print(p.shape)
            for i in range(pred_depth.shape[0]):
                for j in range(pred_depth.shape[1]):
                    # print(al_hit*p_hit(pred_depth[i,j], true_depth[i,j], sigma))
                    p[i,j] = al_rand*self.p_rand(pred_depth[i,j], self.z_max) + al_hit*self.p_hit(pred_depth[i,j], true_depth[i,j], sigma) + al_max*self.p_max(pred_depth[i,j], self.z_max)
                    # print(p[i,j])
            # print(p)

            x = np.arange(0,100,1)
            
            # plt.figure()
            # plt.hist(pred_depth.flatten(), 100)
            # plt.plot(pred_depth.flatten(), p.flatten()*100)
            # # plt.plot(x, self.p_hit(x, 2, 10))
            # plt.show()

            return al_hit, al_max, al_rand, self.sigma