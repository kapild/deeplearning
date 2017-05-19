
# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

--

### Writeup

#### 1. [Here]() is the writeup for project.

You're reading it!

- All the code is in Python notebook `P4_Advanced_Lane_Detection.ipynb`

### 1. Camera Calibration
The first steps to identify lane detection is to perform Camera calibration. This helps in un-distorting the image from a distorted camera. 

#### How to calibrate an Camera:
OpenCV provides and API to calibrate a Camera.

`cal_dict = calibrate_camera(objpoints, imgpoints, cal_image)`

- It requires object points and image points and an image for calibration. 

Thus, the goal of finding `objpoints ` and `imgpoints ` is implemented in 2 sections of the python notebook. 
 
- ( `#1.-Find-corners-of-Chessboard`) 
- (` #2.-Calibrate-and-UnDistort-image`) 

The basic idea is to first find the corner points of Chessboard and use that to calibrate the camera. 

### 1.1. Find image and object points

- The first step is to create the objects points `objpoints`. This is created using `numpy` and size of the array is `row=9` and `col=6`. This is a 3-D array, which consists of points of a correct chessboards in the real world. Here, the z-axis is `Zero`, since all the points lies flat on z-axis. - Sample data for `objpoints` for the first coloumn is shown below. 

```
	   [ 0.,  0.,  0.],
       [ 1.,  0.,  0.],
       [ 2.,  0.,  0.],
       [ 3.,  0.,  0.],
       [ 4.,  0.,  0.],
       [ 5.,  0.,  0.],
       [ 6.,  0.,  0.],
       [ 7.,  0.,  0.],
       [ 8.,  0.,  0.],
```

- This array is used whenever I sucessfully find the corners of chessboards using the function `findChessboardCorners`
- The `findChessboardCorners` functions retruns me the corners found in the image. These image points are appended in the array `imgpoints`

### Plotting the corners found: 
- For debugging, I plotted the corners found by the function `findChessboardCorners`. Some samples are:

|      		|     	        			| 
|:---------------------:|:---------------------------------------------:| 
| ![Corner 1](./output_images/pics/corners1.png)       | ![Corner 2](./output_images/pics/corners2.png)   |
| ![Corner 3](./output_images/pics/corners3.png)     | ![Corner 4](./output_images/pics/corners4..png)|
| ![Corner 5](./output_images/pics/corners5..png)    | ![Corner 1](./output_images/pics/corners6..png) |

- I used all the 20 images to find `objpoints, imgpoints` from the function:

 ```
 def get_chess_image_point_corner_pts(num_row, num_col, glob_path, is_plot=False, is_debug=False):
 ```

### 1.2. Actual calibration

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. The necessary variables were stored in a dict

```
    cal_dict = {}
    cal_dict['mtx'] = mtx
    cal_dict['dist'] = dist

```

-  I applied this distortion correction to the test image using the function,

```
cv2.undistort(img, cal_dict['mtx'], cal_dict['dist'], None, cal_dict['mtx'])
```
Sample images can be found here :-

![alt text](./output_images/pics/undistort1.png)
![alt text](./output_images/pics/undistort2.png)


- This concludes the first section of Calibrating the camera and findings it undistorted image.


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

From the above step, I was able to compute the correct camera matrix and distortion coefficients. These coefficients were then used apply distortion correction to the test images provided in `test_images` folder. 
- distortion correction was applied using the following function in my python notebook

```
def test_un_distortion(perspective_image_path):
    pesp_img = cv2.imread(perspective_image_path)
    pesp_img_rgb = cv2.cvtColor(pesp_img, cv2.COLOR_BGR2RGB)

    img_un_dist_rgb = un_distort_image(pesp_img_rgb, cal_dict)

    plot_two_images(pesp_img_rgb, img_un_dist_rgb, plot_diff=True)
```
- This functions reads an image from a local path, converts the image to `RGB`, un distorts it using the coefficients.
- For, helpful debugging, I plotted the diff of the image too. The results are as below:-

![alt text](./output_images/test_image_1_undistort.png)
![alt text](./output_images/test_image_2_undistort.png)


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

All code for this this part is in `Section #4` and `Section #5` of the notebook

I applied image gradients and color transfor in section 4 and 5 of the python notebook. Some details of these sections are mentioned below:

### 2.1 Apply image gradients 
- In this section, I basically implemented 3 different functions related to image transform. The function definitions of each functio is as follows:

```
# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def magnitute_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
```

```
# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh_min=0, thresh_max=255):
```

```
Define a function to threshold an image on directon for a given range and Sobel kernel
def direction_threshold(img, sobel_kernel=3, dir_thresh=(0, np.pi/2)):

```

The output from these function can be applied in many ways, but I decided to only implement one. In the implementation, I `and` the output from gradx and grady and also `and` the output from direction and mag gradients. These ouputs were then `ORed`

```
def factory_combine_gradients_mag_direction (grad_x, grad_y, mag_thres, dir_thres, mode=1, is_plot=False):
    combined_binary = np.zeros_like(dir_thres)
    if mode == 1:
        return combine_and_x_y_or_and_mag_dir(grad_x, grad_y, mag_thres, dir_thres, is_plot=is_plot)
    raise Exception("Invalid mode:" + str(mode))
    
```
The output from each of these functions can be seen here

![image graients](./output_images/apply_image_gradients.png)


### 2.2 Apply HLS Color spaces 
- Then I looked into how the images looked into `HLS` and `RGB` color space. 
- After doing visual inspection,  I applied gray scale thresholding on `R`, `H`m  `L` and `S` channel. 
- Finally, I settled with `H` channel only  

|   HLS RGB images   		|     	        H,S,R thresholding			| 
|:---------------------:|:---------------------------------------------:| 
| ![image graients](./output_images/pics/hls_rbg_channels.png)     |![image graients](./output_images/pics/hls_threshold.png)   |


### 2.3 Combine Color threshold and gradient threshold
- Finally, in section 4.3. I combined the results from above two sections to produce a single image. 
- After various attemps, I only picked `grad x` thresholdings and `S` channel thresholing from HLS channel. In future work, I will try to combine all the channels for better image detection. But, for now, this worked fine. 
 - This is implemented in `# 4.2 Combine Color threshold and gradient threshold`
![alt text](./output_images/combined_grad_x_s_select.png)

The above functions are implemented in a form of pipeline:

```
# Edit this function to create your own pipeline.
def pipeline(img_BGR, s_thresh=(170, 255), sx_thresh=(20, 100), is_plot=False, add_debug_image=False):
    
```


### Final binary image
![Final](./output_images/final_binary.png)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

- The code for perspective transform is in the section `# 3. Step to perform Perspective Transform`
- To apply perspective transform we need pair of 4 points. These points are of an trapezoid in the original image and rectangle points where we would like to transform the image. 

The function definition is:

```
def get_perspective_transform(src, dest, undist_image):
    img_size = (undist_image.shape[1], undist_image.shape[0])
    M = cv2.getPerspectiveTransform(src, dest)
    warped = cv2.warpPerspective(undist_image, M, img_size, flags=cv2.INTER_LINEAR)
    M_INV = cv2.getPerspectiveTransform(dest, src)
    un_warped = cv2.warpPerspective(warped, M_INV, img_size, flags=cv2.INTER_LINEAR)
    
    return M, M_INV, warped, un_warped
```

- I picked the following points

```python
src_pnts = np.float32([[585, 450], [203, 720], [1127, 720], [695, 450]])
dst_pnts = np.float32([[320, 0], [320, 720], [960,720], [960, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 450      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 450      | 960, 0        |

- Using the `src` and `dst` points , `get_perspective_transform` gave me both the wrapped and unwrapped image.
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
- The source points are joined with drawline function from Open CV.
- I also plotted the inverse of the wrap image, to understand how the unwrapped image looks.

![](./output_images/persp_transform_4.png)
![](./output_images/persp_transform_2.png)
![](./output_images/persp_transform_3.png)


## Pipleine of undistortion, thresholding and prespetive transform. 
- Finally, I combined the output from all of the sections above to get the final image. This image will be used to calculate the radius of curvature and fitting of polynomial of the lane detection.

- The output for a single image looked something like this:-

|  Test image 2 pipeline  		|     	          Test image 5 pipeline			| 
|:---------------------:|:---------------------------------------------:| 
| ![image graients](./output_images/pipeline_2.png)     |![image graients](./output_images/pipeline_1.png)   |


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
For line detection, I started with the simple approach of finding historgams in the bottom part of the image. 

The code is defined in `Section 7` of my notebook. The steps to find lane pixels and fitting polynomials are described below.

## 4.1 Window search
- `Histogram`: Firstly, I take the histogram of the bottom half of the transformed binary image generated from the above section. This works well since lanes are usually parallel and vertical to begin with. 
- `Max points`: Secondly, since there are two lanes, I take a midpoint of width of the image and find the maximum point on both side of the images. These points are starting point of where my alogorithm thinks I have lane edges. 
- `Window search`: With the `max points` as starting points, I used window search to search for the lanes. There are few parameters used to search for window 'nwindows` and `margin`. The first parameter decides the height of the window whereas the second defines the width. From the left and right maxpoint, a window of height = (image_height/nwindows) and width = margin, is used to search for non zero points. These points are stored. If num of points searched are found over a threshold, the next search is centered at the median of these points. 
- Finally, after `nwindows` loop, I have list of indecies for each window where pixel value is non zeros. 
- This gives me set of points, where I believe there are left and right lanes. The output looks something like this. 

Sample code looks like this

```
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
```

## 4.2 Fit Polynomial degree 2.
- After detecting the points as lanes, the next task is to fit the second order degree polynomial to left and right side of image.
- This is easy as I have x and y points of pixels for both left and right side of image. 

The code looks like this

```

def fit_ploy_2_degree(leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit
```
Here are three images from histogram, window search and fitting polynomials.


|  Histogram  		|  Window search			| Fit Poly with lines in white|
|:---------------------:|:---------------------------------------------:| :---------------------------------------------:| 
| ![image graients](./output_images/detect_poly_hist.png)     |![image graients](./output_images/detect_poly_window_search.png)   |![image graients](./output_images/detect_poly_fit_line.png) 

## 4.2 Search for lanes next frame. 

- Once the lanes are identified, one doesn't need to do exhaustive window search to find lanes in next frame again. 
- Finding the lanes in next frame is easy: Just look for lanes in the already fitted polynomial from previous images. 
- This is implemented in Section `## 7.3 Find polynomial fit of image using already fitted polynomial fit from previous image`. This is how it looked with the search area marked in green.

![](./output_images/detect_poly_next_frame.png)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius of curvature was calculated in section 7.4. `Calculate Radius of curvature`. 
- It takes the coeffecients of left and right fitted poly and gets the radius of curavature using the derivative and double derivate of the fit function.
- However, one this to note here is that the radius is calculated pixels image. To convert the radius of curvature in real world dimension, we need to convert the pixels to real world space. This is done by the following. 

```
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

- The final pipeline is plotted in section of the notebook. `# 8. Final pipeline of detection and plotting`

![](./output_images/final_image_wrap.png)
![](./output_images/final_image_uwrap.png)
![](./output_images/final_image.png)


- For some helpful debugging, I ploted images from all the pipelines in one images. This is how it looks

![](./output_images/debug_images_2.png)

![](./output_images/debug_images_1.png)


--

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_submission_final.mp4)

[Debug video](./project_video_submission.mp4): Here is a link to my debug video which comprises of all the images from the pipeline. It helped me a lot in tweaking the parameters where model was not performing well.


--

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

### Current approah:
- For color thresholding I only used S channel for now. 
- For grdaient thresholding I only used `x` gradient. 
- For ploy fitting, I only used window search followed by polly fit on the next frame. 

### Debug video
The most important thing which helped me getting the best output ws debugging video. 
- My initial margin for detecting lanes was `100`. This was causing issue when there were shadows. The detection was falsely picking the shadows as the non zero pixels causing to change the means. Ths resulted in incorrect detection. 
- The AHA moment came by looking at the debug video. By using  the margin as only `30`, the detection algo. only searched very close to only detected lines and didn't identify shadows.

- Link is here [Debug video](./project_video_submission.mp4)

### Future work 
Since, the pipeline does a bad job on harder videos, I plan to work on the following changes. 

- addtion to HLS and RGB color look at other channels and use them for thresholding. 
- Instead of combining just S channel and grad x sobel, combined other channels as well.
- Look into smoothing the ploynomial fit 