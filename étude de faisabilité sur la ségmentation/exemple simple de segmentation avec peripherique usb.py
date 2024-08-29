from ultralytics import FastSAM

model = FastSAM('FastSAM-s.pt')

result=model.track(source=2, imgsz=640, save=False, show=True)
