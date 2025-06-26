# ��Ŀ���ƣ�CamLIME

����ĿΪ���� **�������༤��ľ��������ֲ����ͷ����о���**�����ܶ���2025��������������Ĺٷ�ʵ�֡������ṩ��������ʵ����롢�������������Լ��������̣�����ѧ������������о���

> �������ӣ�[arXiv / IEEE / ACM / Springer ����]  
> ��ϵ���ߣ�[1041674453@qq.com]  
> ���ߵ�λ��[RAier/�������ѧ�빤��ѧԺ/��������ѧ]

---

## ��Ŀ�ṹ˵��

```bash
CamLIME/
������ src/                      # ���Ĵ���ʵ��
������ scripts/                  # ѵ��/���Խű�
������ configs/                  # �����ļ�
������ data/                     # ���ݼ�����Ԥ���������˵���ĵ���
������ checkpoints/              # Ԥѵ��ģ�ͻ򱣴��Ȩ��
������ requirements.txt          # Python�����б�
������ CAM_LIME.py               # CamLIME���Ĵ���
������ CAM_LIME_w_f.py           # Ȩ���볬����������ϵ����
������ CAM_LIME_fues_act.py      # ����������ںϼ���
������ my_difact.py              # CamLIME����в�ͬ����ģʽ�Ա�Ч��
������ my_diflayer.py            # CamLIME����в�ͬ�����Ա�Ч��
������ my_difact_diflayer.py     # CamLIME����в�ͬ����ģʽ����ͬ�����˫�����Ա�Ч��
������ my_difmodel.py            # CamLIME��ܶԲ�ͬ�ں�ģ�͵Ľ���Ч��
������ my_difExp_vis.py          # CamLIME�벿�ֽ��ͷ����ĶԱ�
������ my_metrics.py             # ����Quantus��д������ָ��
������ my_Complexity.py          # CamLIME���������ͷ����ĸ��ӶȱȽ�
������ my_Faithfulness.py        # CamLIME���������ͷ�������ʵ�ԱȽ�
������ Quantus_CamLime_all.py    # ��Quantus�����CamLIME������ָ�����
������ README.md                 # ��Ŀ˵���ĵ�
������ LICENSE                   # ��Դ���֤


```

---

## ��������

����Ŀ���� Python �������Ƽ�ʹ�� Anaconda �� Virtualenv ����⻷����

### Python���������� `requirements.txt`����

```txt
python>=3.8
numpy
torch>=1.11.0
torchvision
scikit-learn
matplotlib
tqdm
```

### �������������⻷������ѡ����

```bash
conda create -n yourproject python=3.8
conda activate yourproject
pip install -r requirements.txt
```

---

## ���ٿ�ʼ

### 1. ��¡�ֿ�

```bash
git clone https://github.com/YourUsername/YourProjectName.git
cd YourProjectName
```

### 2. �������ݣ���ʹ��˵����

�뽫���ݼ������� `data/` Ŀ¼�£���ο� `data/README.md` ��ȡ�������ط�ʽ����֯�ṹ��

### 3. ѵ��ģ��

```bash
python scripts/train.py --config configs/your_config.yaml
```

### 4. ����ģ��

```bash
python scripts/test.py --checkpoint checkpoints/model_best.pth
```

---

## ʵ��������

���踴�������еĶ��������ͼ����ο���

- `scripts/eval.py`����������ָ������
- `scripts/plot.py`�����ڻ�ͼչʾ
- ������Դ��Ԥ����˵���� `data/README.md`

---

## ���ñ�����

��������о���ʹ���˱���Ŀ�Ĵ��룬�������������ģ�

```bibtex
@article{yourpaper2025,
  title={Your Paper Title},
  author={Author1 and Author2 and Author3},
  journal={Journal Name},
  year={2025},
  volume={xx},
  number={yy},
  pages={zz-zz},
  publisher={Publisher}
}
```

---

## ��Ȩ�����֤

����Ŀ���� MIT License ��ɣ�������� [LICENSE](./LICENSE) �ļ���

```
MIT License

Copyright (c) 2025 Author

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

---

## �������⣨FAQ��

1. **Q: ���ݼ�����ʧ����ô�죿**  
   A: ��ȷ���������ӻ�ʹ�ù��ھ��񣬻���ϵ���߻�ȡ������ӡ�

2. **Q: ʹ�� GPU ѵ��ʱ����**  
   A: ��ȷ�� CUDA ������ PyTorch �汾ƥ�䡣

3. **Q: ������ģ�ͽṹ�����޸���**  
   A: ���ԣ��Զ���ģ��� `src/models/`��

---

## �����뷴��

��ӭ�ύ Issue �� Pull Request ���й��ף������κ�������飬Ҳ��ӭͨ���ʼ���ϵ���ߡ�
