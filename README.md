# CA2-2B01-Emotive-Web

### Front-End To do:

* History Filter Function
  * Date Range (If got time, else, replicate the dashboard)
* Connect to TF Server to get Prediction
  * Send the black and white (resized?) cropped photo

### General To Do:

* Validation
* Testing
* Code Clean Up

### Errors/Bugs

* Camera with FaceAPI detection is slow
* Camera cannot CORS (i.e. localhost to 127.0.0.1)
* When loading the model, the buttons cannot be pressed
* Camera does not work on Firefox due to incompatible APIs
* Deleting a history item will reset the history to default filters

### Edit in face-api.min.js

1. Change rectangular border colour to theme purple
2. Remove confidence score due to mirroring bug: From `new sv(n,{label:r})` to `new sv(n,{label:""})`
