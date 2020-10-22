<?php
  $user_id = filter(INPUT_POST, 'user_id', FILTER_VALIDATE_INT)
?>
<!doctype html>
<html>
  <head>
    <meta name="viewpoint" content="width='device-width', initial-scale='1'">
    <title "CS">
  </head>
  <body>
    <heading>
      <center> 
        <div class="heading">
          CS Project
        </div>
      </center>
      <div id="welcome-container">
        <div id="welcome-text">
          Your selected user id is <span><?php echo $user_id; ?></span>
        </div>
    </heading>
  </body>
</html>
