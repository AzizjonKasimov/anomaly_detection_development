Anomaly_detection_deployment
Repository for the deployment of anomaly detection module
``` bash
git clone https://github.com/AzizjonKasimov/anomaly_detection_deployment.git
```

Create and switch to a new branch
``` bash
git checkout -b your-name-of-the-branch
```

Verify you are in a new branch
``` bash
git status
```

After making changes add the changes to the stage. Code below is to add files that were changed.
``` bash
git add -A
```

If you want to clear the stage then use 
``` bash
git reset 
```

Sometimes Git's cache needs to be cleared
``` bash
git rm -r --cached .
```

After adding files to the stage commit the changes with a message
``` bash
git commit -m "Updated the files"
```

Push the changes to the branch name
``` bash
git push origin your-name-of-the-branch
```

Note: there are slight variations of the commands above, so try to troubleshoot if you have some problems.

Note: I made so that git will ignore uploading the datasets. Therefore, locate "datasets" folder in GNNAD_training and "production_datasets" into preprocess_data_short.