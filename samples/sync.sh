#!/bin/bash

echo  $(dirname $0)
rsync -av --include '*/' --include '*.png' --include '*.log' --exclude '*' $(dirname $0)/ ~/Dropbox/FukunagaLabShare/OngoingWorks/Asai/latent-planner/sync/
