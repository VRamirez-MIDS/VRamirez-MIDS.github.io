#!/bin/bash

kubectl get virtualservices.networking.istio.io -A | grep mids255.com | awk '{print $4}' | tr -d '["]' | awk -F. '{print $1}' > links.txt
<links.txt xargs -t -I {} bash -c 'NAMESPACE={}; k6 run load.js'
