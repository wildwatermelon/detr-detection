import VIA2COCO.convert as via2coco

first_class_index = 0

data_path = 'D:/ITSS/balloon_dataset/balloon/'

for keyword in ['train', 'val']:

  input_dir = data_path + keyword + '/'
  input_json = input_dir + 'via_region_data.json'
  categories = ['balloon']
  super_categories = ['N/A']
  # can only be custom_train and custom_val, see detr/datasets/custom.py
  output_json = input_dir + 'custom_' + keyword + '.json'

  print('Converting {} from VIA format to COCO format'.format(input_json))

  coco_dict = via2coco.convert(
      imgdir=input_dir,
      annpath=input_json,
      categories=categories,
      super_categories=super_categories,
      output_file_name=output_json,
      first_class_index=first_class_index,
  )