var imageURL = [];
$.getJSON("https://api.unsplash.com/search/photos?client_id=d7f8c928376d63dffc235980d53eaf75d191597018794107b3db91ab4dd314d9&query="+document.getElementById("keyw").value+"&per_page=20&w=400&h=300"
, function(data){
    console.log(data.results);
      for(var i = 0; i < 20;i++){
        console.log(data.results[i].urls.regular);
        console.log(data.results[i].id);
        answer = data.results[i].urls.regular;
        imageURL[i] = answer;
      }
    // $.each(data, function(index, value) {
    //     console.log(data.results);
    //    // imageURL.push(value[1]);
    //
    // $('#output').append('<a>' + imageURL + '<br><br><br></a>');
    //    });
});
function myFunction(){
      var len = 20;
      for(var j = 0;j < len;j++){
        document.getElementById("output"+j).src = imageURL[j];
      }
}
// &orientation=landscape
// Can be added to api Unsplash Query
//https://api.unsplash.com/users/annak/likes?client_id=68b53e18cc7fcb84ac340f31b86995d5a82de6bc7cc8312a5aa4d24a6ab04c68
