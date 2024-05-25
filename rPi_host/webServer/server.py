import app

##Main
if __name__ == '__main__':
    #Creates a multiprocessing object and runs it's loop
    app.image_save.value = True
    app.p.start()
    try:
        app.web.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False) #Web server launch
    except KeyboardInterrupt:
        app.p.stop() #Stops the inference loop
