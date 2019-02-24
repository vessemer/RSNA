# RSNA Pneumonia Detection challenge
#### 40th place solution [silver]

The lungs segmentation via UNet model were performed first to take a closer look on data distribution.
![Lungs segmentation](https://habrastorage.org/webt/nb/za/mx/nbzamx6cxtqj3yrhoxr3egzja-e.png)

Resulted ROI were used then as describtors for clusterisation:  
![Clusters](https://habrastorage.org/webt/6k/if/qw/6kifqwxfehnqes4styicxq-mp90.png)

This two resulted clusters actually caught the difference of images came from different sources.
Below are random samples from each of both clusters:  
![Samples](https://habrastorage.org/webt/yf/ie/hm/yfiehmgrooywr39nknegwgcdu78.png)

To overcame the domain difference an augmentation technique based on work of [Philipsen et al](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7073580&isnumber=7229384) has been employd [`enorm.py`](https://github.com/vessemer/RSNA/blob/671e4077b23b81a77f566d73944bc105406e304d/utils/enorm.py):
![Energy augmentation](https://habrastorage.org/webt/cy/yt/fa/cyytfahd_wzgz-tfbepymfwkdhg.png)

Finally RetinaNet model was used along with focal loss:  
![Loss plot](https://habrastorage.org/webt/mb/jj/7u/mbjj7upwwnflxkf-moezywavbw4.png)
