import { Component } from '@angular/core';
import * as math from 'mathjs';
import * as nerdamer from 'nerdamer';
import * as tf from '@tensorflow/tfjs';
import * as _ from 'lodash';
import {DataFrame} from "dataframe-js";

const ReservedParams = ['a','b','c','d','e',"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda"];

const arrSum = function(arr){
  return arr.reduce(function(a,b){
    return (+a) + (+b)
  }, 0);
}

class tfGraph {

  variables:string[] = [];
  formula_head;
  y;
  x_input;
  model;
  lossAndOptimizer;
  varDic = {};


  constructor(private mathFormulaInput:string){
    // console.log( math.parse( mathFormulaInput));
    this.formula_head = math.parse( mathFormulaInput);
    // this.collect_variables();
    // console.log(this.varDic);
    this.allocate_variables();
    // console.log(this.varDic);
    // this.y = this._rec_tree2tf(this.formula_head);
    // console.log( 'graph', this.y);
    // this.build_model();
  }

  // _rec_tree2tf ( node, ) {
  //   let args = node.args;
  //   for( let i = 0; args && i < args.length; i++){
  //     args[i] = this._rec_tree2tf( args[i]);
  //   }

  //   switch (node.type) {
  //     case 'OperatorNode':
  //     if ( node.isBinary()){
  //       // console.log(node.fn.slice(0,3), args[0], args[1]);
  //       return tf[node.fn.slice(0,3)]( args[0], args[1]); //taking the prefix of the operator to match tf methods naming
  //     }
  //     else{
  //       return tf.mul(-1, args[0]); //assuming unary operator => unary minus
  //     }
  //     case 'ConstantNode':
  //     // console.log(node.type, node.value);
  //     return tf.variable( tf.scalar(node.value), false);
  //     case 'SymbolNode':
  //     // console.log( node);
  //     if (node.name.toLowerCase() === 'exp'){ //exp operator call
  //       return tf.exp( args[0]);
  //     }
  //     else if (node.name.toLowerCase() === 'log'){ //log operator
  //       return tf.log( args[0]);
  //     }
  //     else if (node.name.toLowerCase() === 'x'){ //x input variable -- maybe store it separately to build model(input, output)
  //       // this.x_input = tf.input( {shape:[1,], name:'x'});
  //       // this.x_input = tf.variable( tf.scalar(0), false, node.name ); //giving initial value in the future
  //       // console.log( 'defined x', this.x_input)
  //       return this.x_input;
  //     }
  //       return tf.variable( tf.scalar(0), true, node.name ); //giving initial value in the future
  //     case 'FunctionNode':
  //     // console.log('function node', node.type, node.name, node['fn']['name'], tf[node['fn']['name']] !== undefined)
  //       return tf[node.name]( args[0]); //assuming unary operation
  //     case 'ParenthesisNode':
  //       return this._rec_tree2tf(node.content);
  //     default:
  //     console.log('Other node', node)
  //   }
  // };


  _rec_pred( node, ) {

    switch (node.type) {
      case 'OperatorNode':
      if ( node.isBinary()){
        // console.log(node.fn.slice(0,3), args[0], args[1]);
        // console.log(node.fn, this._rec_pred(node.args[0]), this._rec_pred(node.args[1])); //taking the prefix of the operator to match tf methods naming
        // console.log( node)
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



  pred( x){
    this.x_input = tf.tensor1d( x, 'float32');
    // console.log( this.x_input)
    return this._rec_pred(this.formula_head)
  }

  loss(x,y){
    const pred = this.pred(x);
    return pred.sub( tf.tensor1d(y, 'float32')).square().mean();
  }

  train(x,y){
    let optimizer = tf.train.adam(1e-2);
      // for( let t =0; t < 2e1; t++){
      for( let t =0; t < 2e3; t++){
      optimizer.minimize(() => {
        return this.loss(x,y);
      });
    }
  }

  printValues(){
    for( let name of Object.keys(this.varDic)){
      console.log( name, this.varDic[name]);
      this.varDic[name].print();
    }
  }


  // build_model(){
  //   // console.log('building model', this.x_input, this.y)
  //   this.model = tf.model( {inputs:this.x_input, outputs:this.y});
  //   console.log( this.model);
  //   console.log( this.model.summary());
  //   this.lossAndOptimizer = {
  //     loss: 'meanSquaredError',
  //     optimizer: 'adam',
  //   };
  // }


  // collect_variables(){
  //   let variables = [];
  //   this.formula_head.traverse(function (node, path, parent) {
  //     if (node.type === 'SymbolNode') {
  //       variables.push( node.name);
  //     }
  //   });
  //   this.variables = variables.sort();
  //   // console.log( this.variables);
  // }

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

  // async train( x, y){
  //   let tensor_x = tf.tensor1d( x, 'float32');
  //   let tensor_y = tf.tensor1d( y, 'float32');
  //   this.model.compile( this.lossAndOptimizer);
  //   await this.model.fit( tensor_x, tensor_y, {epochs:3});
  // }


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

  constructor(){
    this.chart = _.cloneDeep( OPTION_TEMPLATE);
    // const parser = math.parser();
    // let f = parser.eval('f(x, y) = x^y');
    // console.log(f);
    // console.log( f(2,3));
    // this.tfDummyExample();

    // this.tfDummyExample_withModel();
  }


  // tfDummyExample(){
  //   // generated dummy data
  //   let x_train = tf.tensor1d( [1,2,3,4,5,6,7], "float32");
  //   let y_train = tf.tensor1d( [1,2,3,4,5,6,7], "float32");
  //   let slope = tf.variable( tf.scalar(0), true, 'slope');
  //   let intercept = tf.variable( tf.scalar(0), true, 'intercept');
  //   const optimizer = tf.train.adam();


  //   function loss(prediction) {
  //     return prediction.sub(y_train).square().mean();
  //   }

  //   console.log('training now');
  //   for( let t =0; t < 5e3; t++){
  //     optimizer.minimize(() => {
  //       const pred = slope.mul(x_train).add( intercept);
  //       return loss(pred);
  //     });
  //     // await tf.nextFrame();
  //   }

  //   console.log('training done');
  //   slope.print();
  //   intercept.print();
  //   const pred = loss(slope.mul(x_train).add( intercept));
  //   pred.print();
  // }



  // tfDummyExample_withModel(){
  //   // generated dummy data
  //   let x_train = tf.tensor1d( [1,2,3,4,5,6,7], "float32");
  //   let y_train = tf.tensor1d( [1,2,3,4,5,6,7], "float32");
  //   let slope = tf.variable( tf.scalar(0), true, 'slope');
  //   let intercept = tf.variable( tf.scalar(0), true, 'intercept');
  //   // let x = tf.input( {shape:[1]});
  //   let x = tf.variable( tf.scalar(0), false, 'input');
  //   let y = tf.add(tf.mul(slope, x), intercept);
  //   const optimizer = tf.train.adam();


  //   // function loss(prediction) {
  //   //   return prediction.sub(y_train).square().mean();
  //   // }

  //   // console.log('training now');
  //   // for( let t =0; t < 5etrained++){
  //   //   optimizer.minimize(() => {
  //   //     const pred = slope.mul(x_train).add( intercept);
  //   //     return loss(pred);
  //   //   });
  //   //   // await tf.nextFrame();
  //   // }

  //   // console.log('training done');
  //   // slope.print();
  //   // intercept.print();
  //   // const pred = loss(slope.mul(x_train).add( intercept));
  //   // pred.print();


  //   let model = tf.model( {inputs: x, outputs: y});
  //   console.log( model);
  //   console.log( model.summary());
  //   let lossAndOptimizer = {
  //     loss: 'meanSquaredError',
  //     optimizer: 'adam',
  //   };

  //   model.compile( lossAndOptimizer);
  //   model.fit( x_train, y_train, {epochs:3});
  // }

  mathInputted(){
    // console.log( 'input expression:', this.mathFormulaInput);
    // let nd = nerdamer( this.mathFormulaInput);
    // let f = nd.buildFunction( undefined);
    // console.log( nd.variables());
    // console.log( f);
    // this.graph = undefined; //could force some garbage collection of the graph
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
      this.chart.series = [serie,]
      this.chart.legend.data = ['data',]
      this.chart = _.cloneDeep(this.chart);
      // console.log( this.chart)
      // console.log( this.train_x, this.train_y);
    }
  }

  fit(){
    if( this.train_x && this.train_y){
      // console.log('training...')
      // this.graph.printValues();
      this.isFitting = true;
      this.graph.train( this.train_x, this.train_y);
      // console.log('trained')
      // this.graph.printValues();

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
      this.isFitting = false;
      this.hasFitted = true;

      // this.graph.print
    }
    else{
      console.log('no training data');
    }
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
  }

  //   let observations = this.train_y.map( x => +x);
  //   let predictions = y_fit;
  //   const sum = observations.reduce((a, observation) => a + observation, 0);
  //   const mean = sum / observations.length;

  //   const ssyy = observations.reduce((a, observation) => {
  //     const difference = observation - mean;
  //     return a + (difference * difference);
  //   }, 0);

  //   const sse = observations.reduce((accum, observation, index) => {
  //     const prediction = predictions[index];
  //     const residual = observation - prediction;
  //     return accum + (residual * residual);
  //   }, 0);
  //     console.log( sum, mean, ssyy, sse);

  //   this.Rsquare =  1 - (sse / ssyy);
  // }
}
