search_button = function(){
    var searched_button = document.getElementById('Search_Button_ca');
    // var logo = document.getElementById('logo_cd');
    // logo.addEventListener('click', function(){
    //     window.location.reload();
    // })
    searched_button.addEventListener('click', function(){
        var searched_about = document.getElementById('Search_input_in').value; //받아온 input
        alert(searched_about);
        //모델 입력 및 출력
        /*

        */
        var ouput_category = '';//출력된 카테고리 값
        location.href = "searched_page.html?category="+ouput_category;
    })
}


image_button = function(){
    var searched_button = document.getElementById('camera_image');
    // var logo = document.getElementById('logo_cd');
    // logo.addEventListener('click', function(){
    //     window.location.reload();
    // })
    searched_button.addEventListener('click', function(){
        var searched_about = document.getElementById('Search_input_in');
        alert("이미지");
        
    })
}