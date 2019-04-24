import { Component } from '@angular/core';
import * as math from 'mathjs';
import * as nerdamer from 'nerdamer';
import * as tf from '@tensorflow/tfjs';
import * as _ from 'lodash';
import {DataFrame} from "dataframe-js";

import {BehaviorSubject, interval, Observable} from 'rxjs';
import { map, pairwise, filter, tap, mapTo } from 'rxjs/operators';

const ReservedParams = ['a','b','c','d','e',"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda"];

// calculates the sum of an array
const arrSum = function(arr){
  return arr.reduce(function(a,b){
    return (+a) + (+b)
  }, 0);
}

class tfGraph {

  variables:string[] = [];
  formula_head;
  y_input;
  x_input;
  model;
  lossAndOptimizer;
  varDic = {};
  optimizer = undefined;
  trainingState = undefined;
  _running = false;
  timer = undefined;


  constructor(private mathFormulaInput:string){
    this.formula_head = math.parse( mathFormulaInput);
    this.allocate_variables();
  }

  _rec_pred( node, ) {

    switch (node.type) {
      case 'OperatorNode':
      if ( node.isBinary()){
        return tf[node.fn.slice(0,3)]( this._rec_pred(node.args[0]), this._rec_pred(node.args[1])); //taking the prefix of the operator to match tf methods naming
      }
      else{
        return tf.mul(-1, this._rec_pred(node.args[0])); //assuming unary operator => unary minus
      }
      case 'ConstantNode':
      // console.log(node.type, node.value);
      return tf.variable( tf.scalar(node.value), false);
      case 'SymbolNode':
      // console.log( node);
      if (node.name.toLowerCase() === 'exp'){ //exp operator call
        // console.log( node.args)
        return tf.exp( this._rec_pred(node.args[0]));
      }
      else if (node.name.toLowerCase() === 'log'){ //log operator
        return tf.log( this._rec_pred(node.args[0]));
      }
      else if (node.name.toLowerCase() === 'x'){ //x input variable -- maybe store it separately to build model(input, output)
        // this.x_input = tf.input( {shape:[1,], name:'x'});
        // this.x_input = tf.variable( tf.scalar(0), false, node.name ); //giving initial value in the future
        // console.log( 'defined x', this.x_input)
        return this.x_input;
      }
      return this.varDic[ node.name];
      // return tf.variable( tf.scalar(0), true, node.name ); //giving initial value in the future
      case 'FunctionNode':
      // console.log('function node', node.type, node.name, node['fn']['name'], tf[node['fn']['name']] !== undefined)
      return tf[node.name]( this._rec_pred(node.args[0])); //assuming unary operation
      case 'ParenthesisNode':
      return this._rec_pred(node.content);
      default:
      console.log('Other node', node)
    }
  };




  pred(){
    return this._rec_pred(this.formula_head)
  }

  loss(){
    const pred = this.pred();
    return pred.sub( this.y_input).square().mean();
  }

  train(x,y){
    let optimizer = tf.train.adam(1e-2);
    this.x_input = tf.tensor1d( x, 'float32');
    this.y_input = tf.tensor1d( y, 'float32');
      // for( let t =0; t < 2e1; t++){
      for( let t =0; t < 2e3; t++){
      optimizer.minimize(() => {
        return this.loss();
      });
    }
  }

  printValues(){
    for( let name of Object.keys(this.varDic)){
      console.log( name, this.varDic[name]);
      this.varDic[name].print();
    }
  }


    allocate_variables(){
      // console.log('inside', this.varDic);
      let tempDic = {};
    this.formula_head.traverse(function (node, path, parent) {
      if (node.type === 'SymbolNode' && ReservedParams.includes(node.name.toLowerCase())) {
        // console.log( tempDic, node)
        tempDic[node.name] = tf.variable( tf.scalar(1), true, node.name, 'float32');
      }
    });
    // this.variables = variables.sort();
    this.varDic = tempDic;
    // console.log( this.variables);
  }

    trainStart(x, y, trainingState){
  // trainStart(duration=3000, batch=10000, isTraining){
    this.optimizer = tf.train.adam(1e-2);
    this.x_input = tf.tensor1d( x, 'float32');
    this.y_input = tf.tensor1d( y, 'float32');

    this.trainingState = trainingState;
    this.trainingState.next( 2);
    this._running = true;
    this.timer = setTimeout( () => this.trainStep());
    // setTimeout( () => {this.trainStop(); isTraining.next(false)}, duration);
    // let resolve = () => {this.trainStop();};
    // await new Promise(resolve => setTimeout(resolve, duration));
  }

    trainStep(){
      for( let t =0; t < 50; t++){
     this.optimizer.minimize(() => {
        return this.loss();
      });
    }
      if( this._running){
    this.timer = setTimeout( () => this.trainStep());
  }
  else{
    this.trainingState.next(0);
  }
  }
  
  trainStop(){
    this._running = false;
    clearTimeout( this.timer);
    this.trainingState.next(0);
  }


}


const OPTION_TEMPLATE = {
    // backgroundColor: '#191919', // matching optimum background color
    title: {
        text: ''
    },
    legend: {
        data: []
    },
    tooltip: {
        trigger: 'axis',
        axisPointer: {
            type: 'cross',
            label: {
                backgroundColor: '#6a7985'
            }
        }
    },
    xAxis: {
        name: 'x',
        // data: [],
        splitLine: {
            show: false
        },
        axisTick: {
            alignWithLabel: true,
        },
        axisLabel: {
            show: true,
            formatter: function (value, index) {
                return (value).toFixed(1);
            },
        },
        // min: 0,
    },
    yAxis: {
        name: 'y',
        splitLine: {
            show: false
        },
        axisLabel: {
            formatter: function (value, index) {
                return (value).toFixed(1);
            },
        },
        // min: 0,
    },
    series: []
};

const CHART_SERIE_TEMPLATE = {
    data: [],
    type: 'line',
    smooth: true,
    name: '',
    showSymbol: false,
};


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'angular-curvefitting';
  mathFormulaInput:string = '';
  // variables:string[] = [];
  df = undefined;
  df_columns:string[] = [];
  x_col = '';
  y_col = '';
  train_x:number[];
  train_y:number[];
  graph = undefined;
  chart = undefined;
  hasFitted = false;
  fittedValues = {};
  Rsquare = 0;
  isFitting = false;
  trainingState = new BehaviorSubject<number>(0);
  policy_update_sub = undefined;
  trainButtonMsg:Observable<string>;
  endOfTrainingObs;

  constructor(){
    this.chart = _.cloneDeep( OPTION_TEMPLATE);

    this.trainButtonMsg = this.trainingState.asObservable().pipe( map( x => {
      if (x === 0){
        return 'Fit';
      }
      if (x === 1){
        return 'Stopping...';
      }
      return 'Stop';
    })
    );
    // console.log('setting them up ');
    this.endOfTrainingObs = this.trainingState.pipe( pairwise(), filter( values => values[0] !== 0 && values[1] === 0));
    //update policy vis when training is finished
    this.endOfTrainingObs.subscribe( x => this.updateFitLine());

  }



  mathInputted(){
    this.graph = new tfGraph( this.mathFormulaInput);
    for( let name of Object.keys( this.graph.varDic)){
      this.fittedValues[ name] = this.graph.varDic[name].dataSync();
    }
  }

  async dataUpload(csvFile){
    this.df = await DataFrame.fromCSV( csvFile)
    this.df_columns = this.df.listColumns();
  }

  XYchange(){
    // console.log( this.x_col, this.y_col);
    let df = this.df.toDict();
    if( this.x_col !== ''){
      this.train_x =  <number[]>df[ this.x_col];
    }
    if( this.y_col !== ''){
      this.train_y =  <number[]>df[ this.y_col];
    }
    if( this.x_col && this.y_col){
      let serie = _.cloneDeep(CHART_SERIE_TEMPLATE);
      for( let index = 0; index < this.train_x.length; index++){
        serie.data.push( [this.train_x[index], this.train_y[index]]);
      }  
      serie['name'] = 'data';
      serie['type'] = 'scatter';
      serie['symbolSize'] =  5;
      this.chart.series = [serie,]
      this.chart.legend.data = ['data',]
      this.chart = _.cloneDeep(this.chart);
      // console.log( this.chart)
      // console.log( this.train_x, this.train_y);
    }
  }

  updateFitLine(){
      // adding fit line
      let x_points = tf.linspace( Math.min(...this.train_x), Math.max(...this.train_x), 1000).dataSync();
      let y_points = this.graph.pred( x_points).dataSync();
      // let temp_x = x_points;
      // let temp_y = y_points.dataSync();
      let serie = _.cloneDeep(CHART_SERIE_TEMPLATE);
      for( let index = 0; index < this.train_x.length; index++){
        serie.data.push( [x_points[index], y_points[index]]);
      }  

      serie['name'] = 'fit';
      this.chart.series = [this.chart.series[0], serie,];
      this.chart.legend.data = ['data', 'fit']
      this.chart = _.cloneDeep(this.chart);
      for( let name of Object.keys( this.graph.varDic)){
        this.fittedValues[ name] = this.graph.varDic[name].dataSync();
      }
      this.calculateRsquare();
  }

  fit(){
    // console.log('training...')
    // this.graph.printValues();
    this.isFitting = true;
    this.graph.train( this.train_x, this.train_y);
    // console.log('trained')
    // this.graph.printValues();
    this.updateFitLine();
    this.isFitting = false;
    this.hasFitted = true;
  }

  changeVarValue(key, newValue){
    // console.log( key, newValue, typeof newValue);
    this.graph.varDic[key].assign( tf.scalar(+newValue, 'float32' ));
  }

  calculateRsquare(){
    // goodness of fit estimation through r-square
    let y_fit = this.graph.pred(this.train_x).dataSync();
    let train_y_mean = arrSum(this.train_y)/this.train_y.length;
    let SStot = arrSum(this.train_y.map( x => Math.pow(x - train_y_mean,2)));
    let SSreg = arrSum(y_fit.map( x => Math.pow(x - train_y_mean,2)));
    // console.log( y_fit.length, this.train_y.length, train_y_mean, SStot, SSreg);
    this.Rsquare = SSreg/SStot;
    this.hasFitted = true;
  }


      trainMode(){
      if ( this.trainingState.getValue() === 0){
        this.policy_update_sub = interval(1500).subscribe( x => this.updateFitLine());
        this.graph.trainStart(this.train_x, this.train_y, this.trainingState);
      }
      else{
        this.trainingState.next(1);
        this.policy_update_sub.unsubscribe();
        this.graph.trainStop();
      }
    }


}
