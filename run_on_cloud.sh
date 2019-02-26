gcloud ml-engine jobs submit training training_1epoch3 --module-name=gtzan.gtzan --package-path=./gtzan --job-dir=gs://featurevisualization --region=europe-west1 --config=gtzan/cloudml-gpu.yaml
