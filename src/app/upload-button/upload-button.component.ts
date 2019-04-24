import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';


@Component({
	selector: 'csv-upload-button',
	templateUrl: './upload-button.component.html',
	styleUrls: ['./upload-button.component.css']
})
export class UploadButtonComponent implements OnInit {

	@Output() onFileUpload = new EventEmitter<File>();


	constructor() {}

	ngOnInit() {
	}

	onFileInput(event){
		let uploadedFile = event.target.files[0];
		if (uploadedFile ){
			this.onFileUpload.emit( uploadedFile);
		}
	}

}
