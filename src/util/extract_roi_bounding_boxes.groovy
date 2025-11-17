// Groovy script to extract ROI bounding box coordinates from annotations in QuPath

def imageData = getCurrentImageData()
def server = imageData.getServer()
def filename = server.getURIs()[0].toString().split('/')[-1].replace('.czi', '')

// Get all annotations in the image
def annotations = getAllObjects().findAll { it.isAnnotation() }
def annotationData = []

// Loop through each annotation and print its coordinates
annotations.each { annotation ->
def roi = annotation.getROI()    
    bounds = [roi.getBoundsX(), roi.getBoundsY(), roi.getBoundsWidth(), roi.getBoundsHeight()]
    print("${filename}, ${annotation.getPathClass()}, ${bounds}")
    annotationData << [filename, annotation.getPathClass(), bounds]
}

def csvHeader = ["filename", "path_class", "bounds_x", "bounds_y", "bounds_width", "bounds_height"]
def csvFile = new File("f:/scratch/${filename}_roi_bboxes.csv")

// Save data to CSV using QuPath's file writing capabilities
csvFile.withWriter { writer ->
  writer.writeLine csvHeader.join(',')  // Join header elements with comma
  annotationData.each { data ->
    writer.writeLine data.flatten().join(',') 
  }
}
