import os
import shutil
import json
import random
random.seed(9999)
path_to_train_anns="/path_to_train_jsons"
yolo_train_anns_path="/path_to_yolo_txt_ann_val"
path_to_test_anns="/path_to_val_jsons"
yolo_test_anns_path="/path_to_yolo_txt_ann_val"
no_object_anns=0
no_object_img_names=[]
no_object_per_cent=4 #per cent without bounding boxes
object_in_train_dataset=7473 #number of images assigned to train data
classes_mapping={"Human":0,"Car":1,"Wagon":2,"SignalE":3,"SignalF":3,"FacingSwitchR":4,"FacingSwitchL":4,"FacingSwitchNV":4,"TrailingSwitchR":5,"TrailingSwitchL":5,"TrailingSwitchNV":5}
for file in os.listdir(path_to_train_anns):
    with open(os.path.join(path_to_train_anns,file),"r") as annotation:
        ann_dict=json.load(annotation)
        if len(ann_dict["bb_objects"])==0:
            with open(os.path.join(yolo_train_anns_path,file.strip(".png.json")+".txt"),"w") as yolo_ann:
                yolo_ann.close()
            no_object_anns+=1
            no_object_img_names.append(file)
        else:
            with open(os.path.join(yolo_train_anns_path,file.strip(".png.json")+".txt"),"w") as yolo_ann:
                for object in ann_dict["bb_objects"]:
                    class_id=classes_mapping[object["class"]]
                    x_center=(float(object["x1"])+float(object["x2"]))/2 / float(ann_dict["img_size"]["width"])
                    y_center=(float(object["y1"])+float(object["y2"]))/2 / float(ann_dict["img_size"]["height"])
                    w=(float(object["x2"])-float(object["x1"])) / float(ann_dict["img_size"]["width"])
                    h=(float(object["y2"])-float(object["y1"])) / float(ann_dict["img_size"]["height"])
                    yolo_ann.write(str(class_id)+" "+str(x_center)+" "+str(y_center)+" "+str(w)+" "+str(h)+"\n")
                yolo_ann.close()

for file in os.listdir(path_to_test_anns):
    with open(os.path.join(path_to_test_anns,file),"r") as annotation:
        ann_dict=json.load(annotation)
        if len(ann_dict["bb_objects"])==0:
            with open(os.path.join(yolo_test_anns_path,file.strip(".png.json")+".txt"),"w") as yolo_ann:
                yolo_ann.close()
        else:
            with open(os.path.join(yolo_test_anns_path,file.strip(".png.json")+".txt"),"w") as yolo_ann:
                for object in ann_dict["bb_objects"]:
                    class_id=classes_mapping[object["class"]]
                    x_center=(float(object["x1"])+float(object["x2"]))/2 / float(ann_dict["img_size"]["width"])
                    y_center=(float(object["y1"])+float(object["y2"]))/2 / float(ann_dict["img_size"]["height"])
                    w=(float(object["x2"])-float(object["x1"])) / float(ann_dict["img_size"]["width"])
                    h=(float(object["y2"])-float(object["y1"])) / float(ann_dict["img_size"]["height"])
                    yolo_ann.write(str(class_id)+" "+str(x_center)+" "+str(y_center)+" "+str(w)+" "+str(h)+"\n")
                yolo_ann.close()

random.shuffle(no_object_img_names)
lenght=int(no_object_anns-object_in_train_dataset/100*no_object_per_cent)
dont_add=no_object_img_names[0:lenght]
print(len(dont_add))
with open("dont_include_to_train.txt","w") as dnt:
    for name in no_object_img_names:
        dnt.write(name+"\n")
    dnt.close()





