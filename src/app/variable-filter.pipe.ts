import { Pipe, PipeTransform } from '@angular/core';

const EXCLUDED_SYMB_NAMES = ['exp', 'log', 'x'];

@Pipe({
  name: 'variableFilter'
})
export class VariableFilterPipe implements PipeTransform {

  transform(variables: string[] ): string[] {
    return variables.filter( x => !EXCLUDED_SYMB_NAMES.includes( x.toLowerCase()) );
  }

}
