from mongo_schemas import *

mtemplate = lambda gscripts, gdivs: """
<!DOCTYPE html>
<html lang="en">
    <head>
        <script>
            var url_login = "https://""" + str(cur_host) + """:8008/login"
        </script>    
        """ + str(gscripts) + """
    </head>
    <body>
        """ + str(gdivs) + """
        <div id="myplot"></div><p>
    </body>
</html>
"""