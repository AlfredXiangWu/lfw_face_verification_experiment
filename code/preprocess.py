import dlib
import cv2
import numpy as np
import os
# import matplotlib.pyplot as plt
import shutil
import scipy.misc
import pickle
from scipy.ndimage.interpolation import rotate
import time

# Download from here: http://ufpr.dl.sourceforge.net/project/dclib/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
predictor_path = "./shape_predictor_68_face_landmarks.dat" # Change this to the file path
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

imgs_directory = './input_imgs/'
out_imgs_directory = './preprocessed_imgs/'

ec_mc_y = 48
ec_y = 48
img_size = 144
crop_size = img_size

# FACIAL DETECTION HELPERS --- START 
def _shape_to_np(shape):
    xy = []
    for i in range(68):
        xy.append((shape.part(i).x, shape.part(i).y,))
    xy = np.asarray(xy, dtype='float32')
    return xy

def get_landmarks(img):
    lmarks = []
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    shapes = []
    for k, det in enumerate(dets):
        shape = predictor(img, det)
        shapes.append(shape)
        xy = _shape_to_np(shape)
        lmarks.append(xy)

    lmarks = np.asarray(lmarks, dtype='float32')
    return lmarks

# FACIAL DETECTION HELPERS --- END
# ------------------------------
# FACIAL ALIGNMENT HELPERS --- START

def align(img, f5pt, crop_size, ec_mc_y, ec_y):
    # change f5pt to double
    f5pt = f5pt.astype('float64')
    ang_tan = (f5pt[0,1]-f5pt[1,1])/(f5pt[0,0]-f5pt[1,0]);# % (le_y - re_y) / (le_x - re_x)
    ang = np.arctan(ang_tan) / np.pi * 180;
    # img_rot = scipy.misc.imrotate(img, 1.3, 'bicubic'); # crops on the shape of the source image
    img_rot = rotate(img, ang, reshape=True);
    imgh = img.shape[0] # size(img,1);
    imgw = img.shape[1] # size(img,2);
    # % eye center
    x = (f5pt[0,0]+f5pt[1,0])/2;
    y = (f5pt[0,1]+f5pt[1,1])/2;
    # % x = ffp(1);
    # % y = ffp(2);

    ang = -ang/180*np.pi;

    [xx, yy] = transform(x, y, ang, img.shape, img_rot.shape);
    eyec = np.round([xx, yy]);
    x = (f5pt[3,0]+f5pt[4,0])/2;
    y = (f5pt[3,1]+f5pt[4,1])/2;
    [xx, yy] = transform(x, y, ang, img.shape, img_rot.shape);
    mouthc = np.round([xx, yy]);

    resize_scale = ec_mc_y/(mouthc[1]-eyec[1]);

    img_resize = scipy.misc.imresize(img_rot, resize_scale, 'bicubic');

    res = img_resize;
    eyec2 = (eyec - [img_rot.shape[1]/2, img_rot.shape[0]/2]) * resize_scale + [img_resize.shape[1]/2, img_resize.shape[0]/2];
    eyec2 = np.round(eyec2);
    img_crop = np.zeros([crop_size, crop_size, img_rot.shape[2]]);
    # % crop_y = eyec2(2) -floor(crop_size*1/3);
    crop_y = eyec2[1] - ec_y;
    crop_y_end = crop_y + crop_size - 1;
    # % crop_y
    crop_x = eyec2[0]-np.floor(crop_size/2);
    # % crop_x
    crop_x_end = crop_x + crop_size - 1;
    box = guard([crop_x, crop_x_end, crop_y, crop_y_end], img_resize.shape[0]);
    # % ADDED BY MOHAMED SAMY: to avoid array out of index errors
    if box[1] > img_resize.shape[1]:
        box[1] = img_resize.shape[1]
    img_crop[box[2]-crop_y+1:box[3]-crop_y+1, box[0]-crop_x+1:box[1]-crop_x+1,:] = img_resize[box[2]:box[3],box[0]:box[1],:];

    cropped = img_crop/255.;
    return [res, eyec2, cropped, resize_scale]

def transform(x, y, ang, s0, s1):
    # % x,y position
    # % ang angle
    # % s0 size of original image
    # % s1 size of target image

    x0 = x - s0[1]/2.;
    y0 = y - s0[0]/2.;
    xx = x0*np.cos(ang) - y0*np.sin(ang) + s1[1]/2.;
    yy = x0*np.sin(ang) + y0*np.cos(ang) + s1[0]/2.;

    # xx = x*np.cos(ang) - y*np.sin(ang);
    # yy = x*np.sin(ang) + y*np.cos(ang);
    return [xx, yy]

def guard(x, N):
    x = np.array(x)
    x[x<1] = 1;
    x[x>N] = N;
    return x

# FACIAL ALIGNMENT HELPERS --- END

# img_fn to be replaced with an img after using sockets
def preprocess(img_fn, out_img_fn):
  
  img = cv2.imread(img_fn, 1)
  start = time.time()
  lmarks = get_landmarks(img)
  print('Detection time: {}'.format(time.time() - start))
  start = time.time()
  if lmarks.size:
    ff_lmarks = lmarks[0]
    nose = ff_lmarks[30, :]
    left_eye_x = sum(ff_lmarks[[37, 38, 40, 41], 0])/ 4
    left_eye_y = sum(ff_lmarks[[37, 38, 40, 41], 1])/ 4
    right_eye_x = sum(ff_lmarks[[43,44,46,47], 0])/ 4
    right_eye_y = sum(ff_lmarks[[43,44,46,47], 1])/ 4
    mouse_left = ff_lmarks[48, :]
    mouse_right = ff_lmarks[54, :]
    ffp_x = np.array([left_eye_x, right_eye_x, nose[0], mouse_left[0], mouse_right[0]])
    ffp_y = np.array([left_eye_y, right_eye_y, nose[1], mouse_left[1], mouse_right[1]])
    f5pt = np.array(zip(ffp_x, ffp_y))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # switch it to rgb
    f5pt = np.array(zip(ffp_x, ffp_y))
    [img2, eyec, img_cropped, resize_scale] = align(img, f5pt, crop_size, ec_mc_y, ec_y);

    img_final = scipy.misc.imresize(img_cropped, [img_size, img_size], 'bicubic')
    if img_final.shape[2] > 1:
        gray_image = cv2.cvtColor(img_final, cv2.COLOR_RGB2GRAY)
    print('Alignment time: {}'.format(time.time() - start))

    cv2.imwrite(out_img_fn, gray_image);
    return True
  else:
    return False

def preprocess_batch(imgs_directory, out_imgs_directory):
  files_dir = os.listdir(imgs_directory)
  people_count = len(files_dir)

  for person_dir in files_dir:
    person_images_files = os.listdir(imgs_directory + person_dir + '/')
    if os.path.exists(out_imgs_directory + person_dir):
      shutil.rmtree(out_imgs_directory + person_dir)
    os.makedirs(out_imgs_directory + person_dir)
    for img_fn in person_images_files:
      out_img_fn = out_imgs_directory + person_dir + '/' + img_fn[:-4] + '.bmp'
      img_fn = imgs_directory + person_dir + '/' + img_fn
      if preprocess(img_fn, out_img_fn):
        print('Preproccessed ' + img_fn.split('/')[-1] + ' successfully.\n')
      else:
        print('Cannot preproccess ' + img_fn.split('/')[-1] + '.')

preprocess_batch(imgs_directory, out_imgs_directory)