

import os
import torchio as tio
class Loader:


    def __init__(self,input_dir):
        self.input_dir=input_dir
        self.subject_list=list(os.walk(input_dir))[0][1]

    def get_subjects(self,brain_extraction=False,test_data=False):

        subjects=[]
        i=0
        for subject in self.subject_list:
            subject_path=os.path.join(self.input_dir,subject)
            path_to_t1=self.path_to_modality(subject_path,'t1')

            if not brain_extraction:
                if not test_data:
                    path_to_seg=self.path_to_modality(subject_path,'seg')
                path_to_brain=self.path_to_modality(subject_path,'brain')
                path_to_t2=self.path_to_modality(subject_path,'t2')
                path_to_t1ce=self.path_to_modality(subject_path,'t1ce')
                path_to_flair=self.path_to_modality(subject_path,'flair')
            i+=1
            if not brain_extraction:
                if not test_data:
                    subject = tio.Subject(
                        t1=tio.ScalarImage(path_to_t1),
                        t2=tio.ScalarImage(path_to_t2),
                        t1ce=tio.ScalarImage(path_to_t1ce),
                        flair=tio.ScalarImage(path_to_flair),
                        label=tio.LabelMap(path_to_seg),
                        brain=tio.LabelMap(path_to_brain)
                    )
                else:
                    subject = tio.Subject(
                        t1=tio.ScalarImage(path_to_t1),
                        t2=tio.ScalarImage(path_to_t2),
                        t1ce=tio.ScalarImage(path_to_t1ce),
                        flair=tio.ScalarImage(path_to_flair),
                        brain=tio.LabelMap(path_to_brain)
                    )
            else:
                subject = tio.Subject(
                    t1=tio.ScalarImage(path_to_t1),
                )
            subjects.append(subject)
        return subjects


    def path_to_modality(self,subject_path,modality):
        for file in os.listdir(subject_path):
            if file.split('.')[0].endswith(modality):
                return os.path.join(subject_path,file)