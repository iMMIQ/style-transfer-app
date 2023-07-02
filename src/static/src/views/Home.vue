<template>
  <body>
  <div>
    <h1>风格转换</h1>
    <div id="show-picture"></div>
    <input id="user-photo" accept="image/jpeg, image/png" type="file" @change="handleStyleImageUpload "/>
    <el-divider></el-divider>
    <div id="show-picture2"></div>
    <input accept="image/jpeg, image/png" type="file" @change="handleContentImageUpload"/>
    <el-divider></el-divider>
    <button :disabled="!styleImage || !contentImage" @click="generateStyledImage">生成</button>

    <div v-if="styledImage">
      <h2>转换后的图片</h2>
      <img :src="styledImage" alt="Styled Image"/>
    </div>
  </div>
  </body>


</template>


<script>
import axios from "axios";

export default {
  name: 'Home',
  data() {
    return {
      styleImage: null,
      contentImage: null,
      styledImage: null,
    };
  },
  methods: {
    handleStyleImageUpload(event) {
      this.styleImage = event.target.files[0];
      var reader = new FileReader();
      reader.readAsDataURL(this.styleImage);
      reader.onload = function () {
        var image = document.createElement("img");
        image.width = "400";
        image.src = reader.result;
        var showPicture = document.getElementById("show-picture");
        showPicture.append(image);
      }
    },
    handleContentImageUpload(event) {
      this.contentImage = event.target.files[0];
      var reader = new FileReader();
      reader.readAsDataURL(this.contentImage);
      reader.onload = function () {
        var image = document.createElement("img");
        image.width = "400";
        image.src = reader.result;
        var showPicture = document.getElementById("show-picture2");
        showPicture.append(image);
      }
    },
    generateStyledImage() {
      const formData = new FormData();
      formData.append('style_image', this.styleImage);
      formData.append('content_image', this.contentImage);

      axios.post('http://127.0.0.1:5000/style-transfer', formData, {responseType: 'blob'})
          .then(response => {
            const styledImage = URL.createObjectURL(response.data)
            this.styledImage = styledImage;
            this.$refs.resultImage.onload = () => {
              URL.revokeObjectURL(styledImage);
            };
          })
          .catch(error => {
            console.error('Error generating styled image:', error);
          });
    },
  },
}
</script>

<style>
body {
  background: lightblue;
}

.pic1 {
  position: page;
  top: 100px;
  left: 50px;
}

.pic2 {
  position: fixed;
  top: 40px;
  right: 50px;
}


</style>