# Malaria-CenterNet
+ based on Objects as Points paper and source
+ detect malaria parasites and white blood cells in thick blood smears


## Main results

|    model    | resolution | precision | recall |   f1  | fps |
|:-----------:|:----------:|:---------:|:------:|:-----:|:---:|
| resnet18prm |  384 x 384 |    90.1   |  98.6  | 94.35 | 125 |
| resnet18prm |  512 x 512 |    91.0   |  99.0  | 95.00 | 100 |
| resnet18prm |  600 x 600 |    94.6   |  99.5  | 97.05 |  91 |


## Installation

+ pip install -r requirements.txt
+ bash compile_coco.sh
+ bash compile_nms.sh


## Dataset

malaria dataset are publicly available at:
+ raw: ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/Thick_Smears_150
+ processed: https://drive.google.com/file/d/1JP9egCgDB36MblsgyklQlGlTH-iluOC3/view?usp=sharing


## Running

+ train:
bash cmd_train.sh

+ eval:
bash cmd_test.sh

+ demo
bas cmd_demo.sh


## License

CenterNet (https://github.com/xingyizhou/CenterNet)
