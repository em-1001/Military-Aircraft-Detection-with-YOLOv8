import os

def create_image_names_file(dir):
    with open(dir + '/_annotations.txt', 'r') as f:
        lines = f.readlines()

    image_names = [line.split()[0].rsplit('.', 1)[0] for line in lines if line.strip()]

    with open(dir + '/image_names.txt', 'w') as names_file:
        for name in image_names:
            names_file.write(f"{name}\n")

def create_label_files(dir):
    with open(dir + '/_annotations.txt', 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line == "":
            continue

        parts = line.split()
        image_filename = parts[0]
        boxes = parts[1:]

        image_name, _ = os.path.splitext(image_filename)
        label_filename = dir + '/label/' + image_name + '.txt'

        with open(label_filename, 'w') as label_file:
            for box in boxes:
                box_parts = box.split(',')
                xmin, ymin, xmax, ymax, class_id = box_parts
                # Writing box coordinates and class to label file
                label_file.write(f"{xmin} {ymin} {xmax} {ymax} {class_id}\n")

if __name__ == "__main__":
    create_image_names_file('./train')
    create_image_names_file('./test')
    os.mkdir("./train/label")
    os.mkdir("./test/label")
    create_label_files('./train')
    create_label_files('./test')