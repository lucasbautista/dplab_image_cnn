# -*- coding: utf-8 -*-

pat_desp_image1 = cv2.imread("patron_desp.png")
pat_desp_image2 = cv2.imread("hrect.jpg")

pat_desp = pat_desp_image1

(h,w) = pat_desp.shape[0:2]
tensor,angles,origins = augmented_data(pat_desp,out_size=(279,279),k=2000)


plt.figure(figsize=(10,10))
plt.imshow(pat_desp[:,:,::-1]),plt.title("Patron  %sX%s"%(pat_desp.shape[0:2]));