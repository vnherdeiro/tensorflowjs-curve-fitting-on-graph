export class DefaultDict {
  dict = {};

  constructor(){}


  get( key){
    if( this.dict[key] === undefined){
    	this.dict[key] = [];
    }
    return this.dict[key];
  }

}
