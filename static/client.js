window.addEventListener("load", init)
//var StravaApiV3 = require('strava_api_v3');
//var defaultClient = StravaApiV3.ApiClient.instance;
//
//// Configure OAuth2 access token for authorization: strava_oauth
//var strava_oauth = defaultClient.authentications['strava_oauth'];
//strava_oauth.accessToken = "YOUR ACCESS TOKEN"
//
//var api = new StravaApiV3.AthletesApi()
//
//var callback = function(error, data, response) {
//  if (error) {
//    console.error(error);
//  } else {
//    console.log('API called successfully. Returned data: ' + data);
//  }
//};



function init() {
    const authButton = document.getElementById('authButton')
    authButton.addEventListener("click", authorize)
}

function authorize(){
    window.location.href = 'http://127.0.0.1:5000/login'
}
