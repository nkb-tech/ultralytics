# Compress guide 

------------------  ------------------

В файле `ultralytics/prune.py` находится код для запуска прунинга:

```
param_dict = {
        'model': 'yolov8s-1limon/weights/best.pt',
        'data':'people_dataset.yaml',
        'imgsz': 640,
        'epochs': 10,
        'pretrained':'yolov8s-1limon/weights/best.pt',
        'batch':10,
        'workers': 20,
        'cache': False,
        'device': [0],
        'close_mosaic': 15,
        'project':'',
        'name':'test',
        'exist_ok': True,
        'patience': 30,
        'amp': False,
        'optimizer': 'SGD',
        'conf': 0.3,
        # prune
        'prune_method':'group_taylor',
        'global_pruning': True,
        'speed_up': 3.0,
        'reg': 0.0005,
        'sl_epochs': 500,
        'sl_hyp': '/home/jovyan/Kozlov_KY/new_ultralytics/ultralytics/ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model':None
        
    }
    prune_model_path = compress(copy.deepcopy(param_dict))
    
    finetune(copy.deepcopy(param_dict),  prune_model_path)
```

```
`param_dict` - обычный YOLO-вский конфиг, в котором добавлены параметры prune в самом конце.
Расшифровка параметров приводится:
`prune_method`- метод прунинга [ "random",  "l1", "lamp",  "slim", "group_slim", "group_sl", "group_taylor", "group_norm", "growing_reg"]
`global_pruning` - обрезает веса в каждом модуле до выбранной степени либо глобально по всему графу
`speed_up` - параметр ускорения или во сколько раз хотите сделать меньше gflops у модели, начать можно с 2х
`reg` - параметр шага прунинга (лучше посмотреть сами методы внутри torch_pruning)
`sl_epochs` - количество эпох обучения для sparse_learning
`sl_hyp` - конфиг sparse_learning
`sl_model` - модель sparse_learning 
```

У меня плохо переводилась модель `sl_model` в onnx, возможно пока не нужно это пробовать.


Логика процесса такая: Сначала идет процесс прунинга, затем файнтюн, сейчас опция resume очень криво работает, поэтому советую отдельно прунить, а затем ставить модельку на дообучение, чтобы не потерять весь прогресс.

В файле `ultralytics/ultralytics/models/yolo/detect/compress.py` 
находится основной код, прунинга, который вызывает `prune.py`


В начале файла выбираются слои, которые игнорятся во время прунинга, этот код зависит от выбранной модели, если вы пруните не обычную yolov8, пожалуйста, измените слои на нужные вам.

В файле `c2f_transfer.py` идет замена c2f на с2faster блоки, это в целом можно отключить, если залезть в него.

Удачные параметры для прунинга советую посмотреть в файле `compress.md`, там собраны разные эксперименты 


# Установка

Установка такая же как с обычным ультралитиксом, с поправкой на способ клонирования с mlspace, почитайте инструкцию Паши, чтобы себе все настроить
```
git clone https://gitlab.ai.cloud.ru/manzherok/ultralytics.git
```

```
cd ultralytics
pip install -e .
```

Также можно запустить прунинг и отдельно finetune через CLI:
yolo detect compress ... do_prune: True do_finetune: True

do_prune - запускать прунинг или нет
do_finetune - тюнить запрунненую модель или нет, эта опция нужна отдельно от прунинга, если вы хотите затюнить запрунненую модель, при обычном обучении yolo подгружаются все веса модели, а не запруненные.