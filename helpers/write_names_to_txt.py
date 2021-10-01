train_imgs_path="path_to_train_images"
test_imgs_path="path_to_val/test images"
dnt_names=[]
import os
with open("dont_include_to_train.txt","r") as dnt:
    for name in dnt:
        dnt_names.append(name.strip("\n").strip(".json"))
    dnt.close()
print(dnt_names)

with open("baseline_train.txt","w") as btr:
   for file in os.listdir(train_imgs_path):
      if file not in dnt_names:
          btr.write(train_imgs_path+file+"\n")
   btr.close()

with open("baseline_val.txt","w") as bv:
   for file in os.listdir(test_imgs_path):
       bv.write(test_imgs_path+file+"\n")
   bv.close()
