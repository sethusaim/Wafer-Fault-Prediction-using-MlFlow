# import os


# def find_correct_model_file(self, cluster_number):

#     self.cluster_number = cluster_number

#     self.list_of_files = os.listdir(self.prod_model_dir)

#     for self.file in self.list_of_files:
#         try:
#             if self.file.index(str(self.cluster_number)) != -1:
#                 self.model_name = self.file

#         except:
#             continue

#     self.model_name = self.model_name.split(".")[0]

#     return self.model_name
