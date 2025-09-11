#!/bin/bash
IMAGE_NAME=project
APP_NAME=project

echo "start minikube" 
# start minikube
minikube stop
minikube delete --all
minikube start --force --driver=docker --cpus=4
chown -R $USER $HOME/.minikube; chmod -R u+wrx $HOME/.minikube
eval $(minikube docker-env)

# echo "login to Azure" 
# authenticate to Azure with your @berkeley.edu
# az login --tenant berkeleydatasciw255.onmicrosoft.com
# az account set --subscription="0257ef73-2cbf-424a-af32-f3d41524e705"
echo "login to ACR" 
# login to ACR repository
# su $SUDO_USER -c "az acr login -n w255mids"
az acr login --name w255mids

echo "cd to ${IMAGE_NAME}" 
# move to the app directory
cd ${IMAGE_NAME}

TAG=$(git rev-parse --short HEAD)
sed "s/\[TAG\]/${TAG}/g" .k8s/overlays/prod/patch-deployment-project_copy.yaml > .k8s/overlays/prod/patch-deployment-project.yaml
echo "create Git has tag = ${TAG}" 

# set IMAGE_PREFIX based on the user's Berkeley email address
IMAGE_PREFIX=$(az account show | jq ".user.name" |  awk -F@ '{print $1}' | tr -d "\"" | tr -d "." | tr '[:upper:]' '[:lower:]')

echo "set image prefix = ${IMAGE_PREFIX}" 
echo "build Docker image = ${IMAGE_NAME}:${TAG}"
# build the Docker image with the tag
docker build --platform linux/amd64 -t ${IMAGE_NAME}:${TAG} .

# set the image name with namespace location in ACR and the git hash as the tag
ACR_DOMAIN=w255mids.azurecr.io
IMAGE_FQDN="${ACR_DOMAIN}/${IMAGE_PREFIX}/${IMAGE_NAME}"

echo"set image name = ${IMAGE_FQDN}"
echo "tag and push Docker images"
echo "docker tag = ${IMAGE_NAME}:${TAG} ${IMAGE_FQDN}:${TAG}"
echo "docker push = ${IMAGE_NAME}:${TAG}"
# tag image with FQDN and push to ACR
docker tag ${IMAGE_PREFIX}/${IMAGE_NAME}:${TAG} ${IMAGE_FQDN}:${TAG}
docker push ${IMAGE_FQDN}:${TAG}
docker pull ${IMAGE_FQDN}:${TAG}

# deploy 
echo "deploy to Azure AKS cluster"
# authenticate to the AKS cluster
az aks get-credentials --name w255-aks --resource-group w255 --overwrite-existing
kubectl config use-context w255-aks

echo "stop pods in Azure AKS cluster"
# stops pods if they are running
kubectl delete pods --all -n victorramirez

echo "start pods in Azure AKS cluster"
# starts pods
kubectl kustomize .k8s/overlays/prod
kubectl apply -k .k8s/overlays/prod
kubectl wait --for=condition=ready --timeout=60s pod -l app=${APP_NAME} -n ${IMAGE_PREFIX}

echo "Done"
