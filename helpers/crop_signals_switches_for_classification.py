import os
from PIL import Image
import json
path_to_train_images="path_to_all_images"
path_to_train_anns="path_to_all_annotations"
signals_train_anns_path="path_to_signal_annotation"
signals_train_img_path="path_to_signal_images"
sw_0_train_anns_path="path_to_facing_sw_annotation"
sw_0_train_img_path="path_to_facing_sw_images"
sw_1_train_anns_path="path_to_trailing_sw_annotation"
sw_1_train_img_path="path_to_trailing_sw_images"

switches_dict_0={"FacingSwitchR":0,"FacingSwitchL":1,"FacingSwitchNV":2}
switches_dict_1={"TrailingSwitchR":0,"TrailingSwitchL":1,"TrailingSwitchNV":2}
signals_dict={"SignalE":0,"SignalF":1}
for file in os.listdir(path_to_train_anns):
    with open(os.path.join(path_to_train_anns, file), "r") as annotation:
        ann_dict = json.load(annotation)
        c_s = 0
        c_sw_0 = 0
        c_sw_1 = 0
        for object in ann_dict["bb_objects"]:
            image = Image.open(path_to_train_images + file.strip(".json")).convert("RGB")
            if object["class"] in signals_dict.keys():
                class_id=signals_dict[object["class"]]
                crop = image.crop((int(object["x1"]),int(object["y1"]),int(object["x2"]),int(object["y2"])))
                crop = crop.resize((128,128))
                crop.save(signals_train_img_path + file.strip(".png.json") + str(c_s) + ".png" )
                with open(signals_train_anns_path+ file.strip(".png.json") + str(c_s) + ".txt","w") as annot:
                    annot.write(str(class_id)+"\n")
                annot.close()
                c_s+=1
            if object["class"] in switches_dict_0.keys():
                class_id=switches_dict_0[object["class"]]
                crop = image.crop((int(object["x1"]), int(object["y1"]), int(object["x2"]), int(object["y2"])))
                crop = crop.resize((258, 128))
                crop.save(sw_0_train_img_path + file.strip(".png.json") + str(c_sw_0) + ".png")
                with open(sw_0_train_anns_path + file.strip(".png.json") + str(c_sw_0) + ".txt", "w") as annot:
                    annot.write(str(class_id) + "\n")
                annot.close()
                c_sw_0 += 1
            if object["class"] in switches_dict_1.keys():
                class_id=switches_dict_1[object["class"]]
                crop = image.crop((int(object["x1"]), int(object["y1"]), int(object["x2"]), int(object["y2"])))
                crop = crop.resize((256, 128))
                crop.save(sw_1_train_img_path + file.strip(".png.json") + str(c_sw_1) + ".png")
                with open(sw_1_train_anns_path + file.strip(".png.json") + str(c_sw_1) + ".txt", "w") as annot:
                    annot.write(str(class_id) + "\n")
                annot.close()
                c_sw_1 += 1