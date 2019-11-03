import cv2
import numpy as np
def make_mark(image,points_vec):
    print(points_vec)
    #mask = np.zeros(image.shape,np.uint8) 
    for i in range(len(points_vec)): ##x_min, y_min, x_max, y_max
        for j in range(int(points_vec[i][1]),int(points_vec[i][3])):
            for k in range(int(points_vec[i][2]),int(points_vec[i][4])):
                image[k][j] = 255
    return image

def make_mask(image,points_vec):
    mask = np.zeros(image.shape,np.uint8)
    
    mask = make_mark(mask,points_vec)
    return mask

def read_points_file(file_name,name_of_object,num_of_people):
    points_file = open(file_name,'r')
    points_vec = [] #y_pos,x_pos,y_length, x_length
    line = points_file.readline().replace("\n", "")
    while line:
        temp = line.split(' ')
        if temp[0] ==name_of_object:
            temp = temp[-6:-1]
            temp_value = (float(temp[2])-float(temp[0]))*(float(temp[3])-float(temp[1]))*float(temp[-1])
            temp_list = []
            temp_list.append(temp_value)
            temp_list += temp[-6:-1]
            #temp_list[1] = int(temp_list[1])-10
            #temp_list[2] = int(temp_list[2])-10
            #temp_list[3] = int(temp_list[3])+10
            #temp_list[4] = int(temp_list[4])+10

            points_vec.append(temp_list)
        line = points_file.readline().replace("\n", "")
    points_vec.sort(reverse=True)
    points_vec = points_vec[num_of_people:len(points_vec)]
    #print(points_vec)
    return points_vec

